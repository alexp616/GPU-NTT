// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <cstdlib>
#include <random>
#include <chrono>

#include "ntt_ct.cuh"

#define DEFAULT_MODULUS

using namespace std;
using namespace gpuntt;

#define BLOCKSZ 512
// #define BLOCKSZ 4
// #define BLOCKSZ 2

// int LOGN;
// int BATCH;

// typedef Data32 TestDataType; // Use for 32-bit Test
typedef Data64 TestDataType; // Use for 64-bit Test

int main(int argc, char* argv[])
{
    CudaDevice();

    int device = 0; // Assuming you are using device 0
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
              << std::endl;

    // const int LOGN = 24;
    // const int len = 1 << LOGN;
    // const Data64 p = 0x1ffffffea0000001ULL;
    // const Root<Data64> root = 0x1d93bdf267fa0b83ULL;
    // const int LOGN = 26;
    // const int len = 1 << LOGN;
    // const Data64 p = 0x1ffffffea0000001ULL;
    // const Root<Data64> root = 0x1d69a1224449865fULL;
    const int LOGN = 30;
    const Data64 p = 0x1ffffff980000001ULL;
    const Root<Data64> root = 0x19ee3223f078e4e7ULL;
    Modulus<Data64> mod(p);

    Root<Data64> curr_root = root;
    for (int currlgn = LOGN; currlgn >= 10; --currlgn) {
        
        NTTFactors<Data64> factors(mod, curr_root, 0);
        NTTParametersCT<Data64> params(currlgn, factors);

        int len = 1 << currlgn;
        Data64* arr = (Data64*)malloc(sizeof(Data64) * len);
        for (int i = 0; i < len; ++i) {
            arr[i] = i;
        }

        Data64* cu_arr;
        cudaMalloc(&cu_arr, len * sizeof(Data64));
        cudaMemcpy(cu_arr, arr, len * sizeof(Data64), cudaMemcpyHostToDevice);

        nttct_configuration<Data64> cfg = {
            .n_power = currlgn,
            .ntt_type = FORWARD,
            .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
            .root_table = params.forward_root_of_unity_table,
            .stream = 0};
        
        GPU_CT_NTT_Inplace(cu_arr, curr_root, mod, cfg);
        
        nttct_configuration<Data64> inversecfg = {
            .n_power = currlgn,
            .ntt_type = INVERSE,
            .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
            .root_table = params.inverse_root_of_unity_table,
            .scale_output = true,
            .stream = 0};

        GPU_CT_NTT_Inplace(cu_arr, params.inverse_root_of_unity, mod, inversecfg);

        cudaMemcpy(arr, cu_arr, len * sizeof(Data64), cudaMemcpyDeviceToHost);

        for (int i = 0; i < len; ++i) {
            assert(arr[i] == i);
        }
        curr_root = OPERATOR<Data64>::mult(curr_root, curr_root, mod);
        
        free(arr);
        cudaFree(cu_arr);
    }

    std::cout << "All CT tests passed!" << std::endl;

    return EXIT_SUCCESS;
}