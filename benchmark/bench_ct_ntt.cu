// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ntt_ct.cuh"
#include "bench_util.cuh"

using namespace std;
using namespace gpuntt; 

#define NUMSPERBLOCK 1024
#define THREADSPERBLOCK 512

// typedef Data32 BenchmarkDataType; // Use for 32-bit benchmark
typedef Data64 BenchmarkDataType; // Use for 64-bit benchmark

void GPU_CT_NTT_Forward_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;

    thrust::device_vector<BenchmarkDataType> input_data(ring_size *
                                                        batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      input_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> small_root_of_unity_table(
        NUMSPERBLOCK);

    state.add_global_memory_reads<BenchmarkDataType>(
        (ring_size * batch_count) + (THREADSPERBLOCK * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    Modulus<BenchmarkDataType> mod_data(10000ULL);
    nttct_configuration<BenchmarkDataType> cfg_ntt = {
        .n_power = static_cast<int>(ring_size_logN),
        .ntt_type = FORWARD,
        .shared_memory = 3 * THREADSPERBLOCK * sizeof(BenchmarkDataType),
        .root = 1234,
        .root_table = thrust::raw_pointer_cast(small_root_of_unity_table.data()),
        .mod = mod_data,
        .stream = stream};

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_CT_NTT_Inplace(
                thrust::raw_pointer_cast(input_data.data()),
                cfg_ntt);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_CT_NTT_Forward_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);

void GPU_4STEP_NTT_Inverse_Benchmark(nvbench::state& state)
{
    const auto ring_size_logN = state.get_int64("Ring Size LogN");
    const auto batch_count = state.get_int64("Batch Count");
    const auto ring_size = 1 << ring_size_logN;

    thrust::device_vector<BenchmarkDataType> input_data(ring_size *
                                                        batch_count);
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(ring_size * batch_count),
                      input_data.begin(),
                      random_functor<BenchmarkDataType>(1234));

    thrust::device_vector<Root<BenchmarkDataType>> small_root_of_unity_table(
        NUMSPERBLOCK);

    state.add_global_memory_reads<BenchmarkDataType>(
        (ring_size * batch_count) + (THREADSPERBLOCK * batch_count),
        "Read Memory Size");
    state.add_global_memory_writes<BenchmarkDataType>(ring_size * batch_count,
                                                      "Write Memory Size");
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    // state.collect_loads_efficiency();
    // state.collect_stores_efficiency();
    // state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    Modulus<BenchmarkDataType> mod_data(10000ULL);
    nttct_configuration<BenchmarkDataType> cfg_ntt = {
        .n_power = static_cast<int>(ring_size_logN),
        .ntt_type = INVERSE,
        .shared_memory = 3 * THREADSPERBLOCK * sizeof(BenchmarkDataType),
        .root = 1234,
        .root_table = thrust::raw_pointer_cast(small_root_of_unity_table.data()),
        .mod = mod_data,
        .scale_output = true,
        .stream = stream};

    state.exec(
        [&](nvbench::launch& launch)
        {
            GPU_CT_NTT_Inplace(
                thrust::raw_pointer_cast(input_data.data()),
                cfg_ntt);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(GPU_4STEP_NTT_Inverse_Benchmark)
    .add_int64_axis("Ring Size LogN",
                    {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28})
    .add_int64_axis("Batch Count", {1})
    .set_timeout(1);