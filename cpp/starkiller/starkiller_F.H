#ifndef STARKILLER_F_H
#define STARKILLER_F_H

#include <AMReX_BLFort.H>

#ifdef __cplusplus
#include <AMReX.H>
extern "C"
{
#endif

  void extern_init(const int* name, const int* namelen);

#ifdef __cplusplus
}
#endif

#endif
