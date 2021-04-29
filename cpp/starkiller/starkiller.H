#ifndef STARKILLER_H_
#define STARKILLER_H_

#include <AMReX.H>

#include <burner.H>
#include <conductivity.H>
#include <eos.H>
#include <network.H>


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
    ReactionSystem();

    // destructor
    virtual ~ReactionSystem();

    // initialize class variables
    void init(const int n_cell, const BoxArray& ba,
              const DistributionMapping& dm);

    // initialize state
    void init_state(const Real dens, const Real temp,
                    const Real xhe, const Real end_time,
                    bool const_state = true);

    // compute the solution
    void sol(Vector<MultiFab>& y);

    // compute the rhs
    void rhs(const Vector<MultiFab>& y,
             Vector<MultiFab>& dydt);

    
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
