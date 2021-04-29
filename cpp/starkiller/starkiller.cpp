#ifndef STARKILLER_H_
#include <starkiller.H>
#endif

// initialize variables
void ReactionSystem::init(const int train_size, const BoxArray& ba,
                          const DistributionMapping& dm) {
    // initialize multifabs
    size = train_size;
    state.resize(size);
    for (int i = 0; i < size; i++){
        state[i].define(ba, dm, NSCAL, 0);
        state[i].setVal(0.0);
    }
    
    // initialize Microphysics
    network_init();   // includes actual_rhs_init()
    eos_init();
}

// initialize state
void ReactionSystem::init_state(const Real dens, const Real temp,
                                const Real xhe, const Real end_time,
                                bool const_state) {
    time_scale = 1.0e-6;
    density_scale = dens;
    temperature_scale = temp;

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

    // initial conditions
    const bool const_flag = const_state;

    // find index of he4
    int he_species = 0;
    for (int i = 0; i < NumSpec; ++i) {
        std::string spec_string = short_spec_names_cxx[i];
        if (spec_string == "he4") {
            he_species = i + FS;
            break;
        }
    }
    if (he_species == 0) {
        Abort("ERROR: he4 not found in network!");
    }

    ResetRandomSeed(time(0));

    for (int i = 0; i < size; i++) {
        for (MFIter mfi(state[i], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<Real> state_arr = state[i].array(mfi);

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
    VisMF::Write("plt_train0", state[0]);
    //WriteSingleLevelPlotfile("plt_train", state[0], {"rho"}, geom, 0.0, 0);

}

// compute the solution at time t
void ReactionSystem::sol(Vector<MultiFab>& state_out) {

}

// compute the rhs at time t
void ReactionSystem::rhs(Vector<MultiFab>& dydx) {

}
