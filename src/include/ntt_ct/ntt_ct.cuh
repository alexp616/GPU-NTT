// (C) Ulvetanna Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan
// Paper: https://eprint.iacr.org/2023/1410

#ifndef NTT_CORE_H
#define NTT_CORE_H

#define CC_89 // for RTX 4090

#include "cuda_runtime.h"
#include "nttparameters.cuh"
#include <functional>
#include <unordered_map>

// --------------------- //
// Authors: Alisah Ozcan
// --------------------- //
typedef unsigned location_t;
/*
#if MAX_LOG2_RINGSIZE <= 32
typedef unsigned location_t;
#else
typedef unsigned long location_t;
#endif
*/

namespace gpuntt
{
    template <typename T> struct nttct_configuration
    {
        int n_power;
        type ntt_type;
        int shared_memory;
        Root<T> root;
        Root<T>* root_table;
        Modulus<T> mod;
        bool scale_output;
        cudaStream_t stream;
    };

    template <typename T>
    __host__ void GPU_CT_NTT_Inplace(T* device_inout, nttct_configuration<T> cfg);


} // namespace gpuntt
#endif // NTT_CT_CORE_H
