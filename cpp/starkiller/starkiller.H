#ifndef STARKILLER_H_
#define STARKILLER_H_

#include <AMReX.H>

#include <burner.H>
#include <eos.H>
#include <network.H>

#include <torch/torch.h>


// state indices
constexpr int NSCAL = 4 + NumSpec;
constexpr int DT = 0;
constexpr int RHO = 1;
constexpr int TEMP = 2;
constexpr int RHOE = 3;
constexpr int FS = 4;  // FirstSpec


using namespace amrex;

class ReactionSystem() {
  public:
  
    // constructor
    ReactionSystem() = default;

    // destructor
    ~ReactionSystem() = default;

    // initialize class variables
    void init(const int n_cell, const BoxArray& ba,
              const DistributionMapping& dm);

    // initialize state
    void init_state(const Real dens, const Real temp,
                    const Real xhe, const Real end_time,
                    bool const_state = true);

    // compute the solution
    void sol(Vector<MultiFab>& state_out);

    // compute the rhs
    void rhs(Vector<MultiFab>& dydx);

    
    /* PRIVATE VARIABLES */
    
    Vector<MultiFab> state;
    int size;
    
    // these are defined in 'starkiller.cpp'
    
    // normalization values
    Real time_scale;
    Real density_scale;
    Real temperature_scale;
    Real energy_scale;
}

#endif
