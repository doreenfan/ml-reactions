#include <starkiller.H>
#include <starkiller_F.H>

#include <extern_parameters.H>

#include <AMReX_VisMF.H>

using namespace amrex;

std::string ReactionSystem::probin_file = "probin";

// constructor
ReactionSystem::ReactionSystem() = default;

ReactionSystem::ReactionSystem(const ReactionSystem& src)
{
    size = src.size;
    state.resize(size);
    for (int i = 0; i < size; i++) {
        state[i].define(src.state[i].boxArray(), src.state[i].DistributionMap(), NSCAL, 0);
        MultiFab::Copy(state[i], src.state[i], 0, 0, NSCAL, 0);
    }

    time_scale = src.time_scale;
    density_scale = src.density_scale;
    temperature_scale = src.temperature_scale;
    energy_scale = src.energy_scale;
}

// destructor
ReactionSystem::~ReactionSystem() = default;

// initialize variables
void ReactionSystem::init(const int train_size, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm)
{
    // initialize multifabs
    size = train_size;
    state.resize(size);
    for (int i = 0; i < size; i++){
        state[i].define(ba, dm, NSCAL, 0);
        state[i].setVal(0.0);
    }

    static bool firstCall = true;

    if (firstCall) {
        // initialize the external runtime parameters
        init_extern();

        // initialize network, eos, conductivity
        network_init();   // includes actual_rhs_init()
        eos_init();
        conductivity_init();

        firstCall = false;
    }
}

// initialize extern parameters
void ReactionSystem::init_extern()
{
    // initialize the external runtime parameters -- these will
    // live in the probin the probin
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "reading extern runtime parameters ..." << std::endl;
    }

    const int probin_file_length = probin_file.length();
    Vector<int> probin_file_name(probin_file_length);

    for (int i = 0; i < probin_file_length; i++) {
        probin_file_name[i] = probin_file[i];
    }

    // read them in in Fortran from the probin file
    extern_init(probin_file_name.dataPtr(),&probin_file_length);

    // grab them from Fortran to C++; then read any C++ parameters directly
    // from inputs (via ParmParse)
    init_extern_parameters();

}

// initialize state
void ReactionSystem::init_state(const Real dens, const Real temp,
                                const Real xhe, const Real end_time,
                                bool const_state)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "initializing initial conditions ..." << std::endl;
    }

    time_scale = 1.0e-6;
    density_scale = dens;
    temperature_scale = temp * 10;

    // do an eos call to set internal energy scale
    eos_t eos_state;
    //eos_state.rho = dens;
    //eos_state.T = temp;

    // pick a composition for normalize of Ye = 0.5
    // w/ abar = 12, zbar = 6
    eos_state.abar = 12.0;
    eos_state.zbar = 6.0;
    eos_state.y_e = eos_state.zbar / eos_state.abar;
    eos_state.mu_e = 1.0 / eos_state.y_e;

    // use_raw_inputs uses only abar, zbar, y_e, mu_e for the EOS call
    // instead of setting those from the mass fraction
    eos(eos_input_rt, eos_state, true);

    energy_scale = eos_state.e;

    // output normalization values
    Print() << "density_scale = " << density_scale << std::endl;
    Print() << "temperature_scale = " << temperature_scale << std::endl;
    Print() << "energy_scale = " << energy_scale << std::endl;

    // initial conditions
    const bool const_flag = const_state;

    // find index of he4
    int he_species = 0;
    for (int i = 0; i < NumSpec; ++i) {
        std::string spec_string = short_spec_names_cxx[i];
        if (spec_string == "He4") {
            he_species = i + FS;
            break;
        }
    }
    if (he_species == 0) {
        Abort("ERROR: he4 not found in network!");
    }

    ResetRandomSeed(time(0));

    for (int l = 0; l < size; l++) {
        for (MFIter mfi(state[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<Real> state_arr = state[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (const_flag) {
                    // state is constant, time varies
                    state_arr(i,j,k,DT) = amrex::Random()*end_time;

                    // set density and temperature
                    state_arr(i,j,k,RHO) = dens;
                    state_arr(i,j,k,TEMP) = temp;

                    // mass fractions
                    for (int n = 0; n < NumSpec; ++n) {
                        state_arr(i,j,k,FS+n) = (1.0-xhe) / (NumSpec-1);
                    }
                    state_arr(i,j,k,he_species) = xhe;
                } else {
                    // time is constant / state varies
                    state_arr(i,j,k,DT) = end_time;
                }
            });
        }
    }
    VisMF::Write(state[0], "plt_x0");
    //WriteSingleLevelPlotfile("plt_train", state[0], {"rho"}, geom, 0.0, 0);

}

// Get the solutions at times dt (stored in state)
void ReactionSystem::sol(Vector<MultiFab>& y)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing exact solution ..." << std::endl;
    }

    // initialize y
    y.resize(size);

    for (int i = 0; i < size; i++){
        y[i].define(state[i].boxArray(), state[i].DistributionMap(), NSCAL, 0);
        MultiFab::Copy(y[i], state[i], DT, DT, 1, 0);
    }

    // evaluate the system solution
    for (int l = 0; l < size; l++) {
        for (MFIter mfi(state[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<Real> state_arr = state[l].array(mfi);
            const Array4<Real> y_arr = y[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // construct a burn type
                burn_t state_out;

                // set density & temperature
                state_out.rho = state_arr(i,j,k,RHO);
                state_out.T = state_arr(i,j,k,TEMP);

                // mass fractions
                for (int n = 0; n < NumSpec; ++n) {
                    state_out.xn[n] = state_arr(i,j,k,FS+n);
                }

                // integrate to get the output state
                Real dt = state_arr(i,j,k,DT)*time_scale;
                integrator(state_out, dt);

                // pass the solution values
                y_arr(i,j,k,TEMP) = state_out.T / temperature_scale;
                y_arr(i,j,k,RHOE) = state_out.e / energy_scale;
                for (int n = 0; n < NumSpec; ++n) {
                    y_arr(i,j,k,FS+n) = state_out.xn[n];
                }
                y_arr(i,j,k,RHO) = state_out.rho;
            });
        }
    }
    VisMF::Write(y[0], "plt_y0");
}

// Get the solution rhs given state y
// scaled solution: ys = y / y_scale
// scaled time : ts = t / t_scale
// f = dys/dts = (dy/y_scale) / (dt/t_scale) = (dy/dt) * (t_scale / y_scale)
void ReactionSystem::rhs(const Vector<MultiFab>& y,
                         Vector<MultiFab>& dydt)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing rhs ..." << std::endl;
    }

    // initialize dydt
    dydt.resize(size);

    for (int i = 0; i < size; i++){
        dydt[i].define(y[i].boxArray(), y[i].DistributionMap(), NSCAL, 0);
        dydt[i].setVal(0.0);
    }

    // evaluate the system solution
    for (int l = 0; l < size; l++) {
        for (MFIter mfi(y[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<const Real> y_arr = y[l].array(mfi);
            const Array4<const Real> state_arr = state[l].array(mfi);
            const Array4<Real> dydt_arr = dydt[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // construct a burn type
                burn_t state_in;

                // set density & temperature
                state_in.rho = y_arr(i,j,k,RHO);
                state_in.T = amrex::max(y_arr(i,j,k,TEMP)*temperature_scale, 0.0);

                // mass fractions
                for (int n = 0; n < NumSpec; ++n) {
                    state_in.xn[n] = max(y_arr(i,j,k,FS+n), 0.0);
                }

                // evaluate the rhs
                Array1D<Real, 1, neqs> ydot;
                actual_rhs(state_in, ydot);
                // note ydot is 1-based

                // pass the solution values
                for (int n = 0; n < NumSpec; ++n) {
                    dydt_arr(i,j,k,FS+n) = aion[n]*ydot(1+n) * (time_scale);
                }
                dydt_arr(i,j,k,RHOE) = ydot(net_ienuc) * (time_scale / energy_scale);
                // C++ networks do not have temperature_rhs; only F90 do
                // dydt_arr(i,j,k,TEMP) = ydot(net_itemp) * (time_scale / temperature_scale);
                // instead, compute average d(temp)/dt
                dydt_arr(i,j,k,TEMP) = (y_arr(i,j,k,TEMP)*temperature_scale - state_arr(i,j,k,TEMP))
                    / (state_arr(i,j,k,DT) * time_scale);
                dydt_arr(i,j,k,RHO) = state_in.rho;
            });
        }
    }
    VisMF::Write(dydt[0], "plt_dydt0");
}
