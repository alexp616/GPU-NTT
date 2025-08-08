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

typedef Data64 TestDataType; // Use for 64-bit Test

// int main(int argc, char* argv[])
// {
//     const int LOGN = 11;
//     const Data64 p = 0x3fffc00000000001ULL;
//     const Root<Data64> root = 0x1f21ab52bcdf8219ULL;
//     Modulus<Data64> mod(p);

//     NTTFactors<Data64> factors(mod, root, 0);
//     NTTParametersCT<Data64> params(LOGN, factors);

//     int len = 1 << LOGN;
//     vector<Data64> arr(len);
//     for (int i = 0; i < len; ++i) {
//         arr[i] = i % 10;
//     }

//     Data64* cu_arr;
//     cudaMalloc(&cu_arr, len * sizeof(Data64));
//     cudaMemcpy(cu_arr, arr.data(), len * sizeof(Data64), cudaMemcpyHostToDevice);

//     nttct_configuration<Data64> cfg = {
//         .n_power = LOGN,
//         .ntt_type = FORWARD,
//         .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
//         .root = root,
//         .root_table = params.forward_root_of_unity_table,
//         .mod = mod,
//         .stream = 0};
    
//     GPU_CT_NTT_Inplace(cu_arr, cfg);
    
//     nttct_configuration<Data64> inversecfg = {
//         .n_power = LOGN,
//         .ntt_type = INVERSE,
//         .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
//         .root = params.inverse_root_of_unity,
//         .root_table = params.inverse_root_of_unity_table,
//         .mod = mod,
//         .scale_output = true,
//         .stream = 0};

//     GPU_CT_NTT_Inplace(cu_arr, inversecfg);
    
//     vector<Data64> arr2(len);
//     cudaMemcpy(arr2.data(), cu_arr, len * sizeof(Data64), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < len; ++i) {
//         assert(arr[i] == arr2[i]);
//     }

//     cudaFree(cu_arr);

//     std::cout << "All CT tests passed!" << std::endl;

//     return EXIT_SUCCESS;
// }

int main(int argc, char* argv[])
{
    CudaDevice();

    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Maximum Grid Size: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
              << std::endl;

    const int LOGN = 30;
    // const Data64 p = 0x1ffffff980000001ULL;
    // const Root<Data64> root = 0x19ee3223f078e4e7ULL;

    const Data64 p = 0x3fffc00000000001ULL;
    const Root<Data64> root = 0x0999e963edecb966ULL;
    Modulus<Data64> mod(p);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 12345);

    Root<Data64> curr_root = root;
    for (int currlgn = LOGN; currlgn >= 10; --currlgn) {
        NTTFactors<Data64> factors(mod, curr_root, 0);
        NTTParametersCT<Data64> params(currlgn, factors);

        int len = 1 << currlgn;
        vector<Data64> arr(len);
        for (Data64& num : arr) { num = dist(gen); }

        Data64* cu_arr;
        cudaMalloc(&cu_arr, len * sizeof(Data64));
        cudaMemcpy(cu_arr, arr.data(), len * sizeof(Data64), cudaMemcpyHostToDevice);

        nttct_configuration<Data64> cfg = {
            .n_power = currlgn,
            .ntt_type = FORWARD,
            .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
            .root = curr_root,
            .root_table = params.forward_root_of_unity_table,
            .mod = mod,
            .stream = 0};
        
        GPU_CT_NTT_Inplace(cu_arr, cfg);
        
        nttct_configuration<Data64> inversecfg = {
            .n_power = currlgn,
            .ntt_type = INVERSE,
            .shared_memory = 3 * BLOCKSZ * sizeof(Data64),
            .root = params.inverse_root_of_unity,
            .root_table = params.inverse_root_of_unity_table,
            .mod = mod,
            .scale_output = true,
            .stream = 0};

        GPU_CT_NTT_Inplace(cu_arr, inversecfg);
        
        vector<Data64> arr2(len);
        cudaMemcpy(arr2.data(), cu_arr, len * sizeof(Data64), cudaMemcpyDeviceToHost);

        for (int i = 0; i < len; ++i) {
            assert(arr[i] == arr2[i]);
        }
        curr_root = OPERATOR<Data64>::mult(curr_root, curr_root, mod);

        cudaFree(cu_arr);
    }

    std::cout << "All CT tests passed!" << std::endl;

    return EXIT_SUCCESS;
}