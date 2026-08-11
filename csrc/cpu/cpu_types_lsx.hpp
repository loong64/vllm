#ifndef CPU_TYPES_LSX_HPP
#define CPU_TYPES_LSX_HPP

#if !defined(__loongarch_sx)
  #error "cpu_types_lsx.hpp requires -mlsx or -mlasx"
#endif

#include "cpu_types_loongarch_defs.hpp"
#include "cpu_types_loongarch_impl.hpp"

#endif  // CPU_TYPES_LSX_HPP
