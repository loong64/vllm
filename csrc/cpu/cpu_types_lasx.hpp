#ifndef CPU_TYPES_LASX_HPP
#define CPU_TYPES_LASX_HPP

#if !defined(__loongarch_asx)
  #error "cpu_types_lasx.hpp requires -mlasx"
#endif

#include "cpu_types_loongarch_defs.hpp"
#include "cpu_types_loongarch_impl.hpp"

#endif  // CPU_TYPES_LASX_HPP
