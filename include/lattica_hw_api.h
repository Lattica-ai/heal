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
#include "permute.h"         // Permutations
#include "shape_manipulation.h"     // Shape manipulation functions

#endif // LATTICA_HARDWARE_API_H
