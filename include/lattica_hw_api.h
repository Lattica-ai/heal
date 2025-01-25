#ifndef LATTICA_HARDWARE_API_H
#define LATTICA_HARDWARE_API_H

// ============= Memory management =============== //
#include "nested_vector.h"  // Host data format
#include "device_memory.h"  // Device data format

// ============= Modular arithmetic ============== //
#include "modsum.h"
#include "modmul.h"
#include "axis_modsum.h"

// ============ Special-purpose ops ============== //
#include "g_decomposition.h" // Gadget decomposition
#include "ntt.h"             // NTT and INTT
#include "permute.h"         // Permutations

#endif // LATTICA_HARDWARE_API_H
