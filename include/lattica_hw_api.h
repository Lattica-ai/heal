#ifndef LATTICA_HARDWARE_API_H
#define LATTICA_HARDWARE_API_H

// ============= Memory management =============== //
#include "device_memory.h"  // Device data format
#include "memory_virtual_ops.h"     // Memory operations
#include "contiguous.h"      // Contiguous memory

// ============= Modular arithmetic ============== //
#include "modop.h"
#include "axis_modsum.h"

// ============ Special-purpose ops ============== //
#include "g_decomposition.h" // Gadget decomposition
#include "ntt.h"             // NTT and INTT
#include "take_along_axis.h" // Permutations
#include "set_const_val.h"   // Set constant value
#include "modmul_axis_sum.h" // Modular multiply and sum

#endif // LATTICA_HARDWARE_API_H
