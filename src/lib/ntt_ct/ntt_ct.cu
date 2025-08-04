#include "ntt_ct.cuh"
// #include <iostream>
#include <vector>

#define LG2THREADSPERBLOCK 9
#define LG2NUMSPERBLOCK 10
#define NUMSPERBLOCK 1024

namespace gpuntt {

    template <typename T>
    __device__ void CooleyTukeyUnit(T& U, T& V, const Root<T>& root,
                                    const Modulus<T>& modulus)
    {
        T u_ = U;
        T v_ = OPERATOR_GPU<T>::mult(V, root, modulus);

        U = OPERATOR_GPU<T>::add(u_, v_, modulus);
        V = OPERATOR_GPU<T>::sub(u_, v_, modulus);
    }

    template <typename T>
    __device__ void GentlemanSandeUnit(T& U, T& V, const Root<T>& root,
                                       const Modulus<T>& modulus)
    {
        T u_ = U;
        T v_ = V;

        U = OPERATOR_GPU<T>::add(u_, v_, modulus);

        v_ = OPERATOR_GPU<T>::sub(u_, v_, modulus);
        V = OPERATOR_GPU<T>::mult(v_, root, modulus);
    }

    template <typename T>
    __device__ T power_mod(T base, int p, const Modulus<T>& m) {
        T result = 1;

        while (p > 0) {
            if (p & 1) {
                result = OPERATOR_GPU<T>::mult(result, base, m);
            }
            base = OPERATOR_GPU<T>::mult(base, base, m);
            p >>= 1;
        }
        
        return result;
    }

    __device__ int bit_reverse(int index, int n_power)
    {
        int res_1 = 0;
        for (int i = 0; i < n_power; i++)
        {
            res_1 <<= 1;
            res_1 = (index & 1) | res_1;
            index >>= 1;
        }
        return res_1;
    }

    template <typename T>
    __global__ void ForwardCore(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Root<T> root, Modulus<T> mod, int glob_stride, int block_stride) {
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);
        
        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x] = root_of_unity_table[threadIdx.x];
        
        int blockStart = blockIdx.x + blockIdx.y * block_stride;
        int globidx1 = blockStart + threadIdx.x * glob_stride;
        int globidx2 = globidx1 + blockDim.x * glob_stride;

        int sharedidx1 = threadIdx.x;
        int sharedidx2 = threadIdx.x + blockDim.x * blockDim.y;
        shared_memory[sharedidx1] = polynomial_in[globidx1];
        shared_memory[sharedidx2] = polynomial_in[globidx2];

        int t_ = LG2NUMSPERBLOCK - 1;
        int t = 1 << t_;
        
        for (int i = 0; i < LG2NUMSPERBLOCK; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x;
            int idx2 = idx1 + t;

            CooleyTukeyUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t >> 1;
            t_ -= 1;
        }
        
        int power1 = blockIdx.x * bit_reverse(sharedidx1, LG2NUMSPERBLOCK);
        int power2 = power1 + blockIdx.x;
        T twiddle1 = power_mod(root, power1, mod);
        T twiddle2 = power_mod(root, power2, mod);
        __syncthreads();
        shared_memory[sharedidx1] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx1], twiddle1, mod);
        shared_memory[sharedidx2] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx2], twiddle2, mod);

        polynomial_in[globidx1] = shared_memory[sharedidx1];
        polynomial_in[globidx2] = shared_memory[sharedidx2];

        return;
    }

    template <typename T>
    __global__ void ForwardCore1(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Root<T> root, Modulus<T> mod, int lg2fftsize) {
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);
        
        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x + threadIdx.y * blockDim.x] = root_of_unity_table[threadIdx.x + threadIdx.y * blockDim.x];

        int blockstart = blockIdx.x * NUMSPERBLOCK;
        int globidx1 = (threadIdx.x + threadIdx.y * blockDim.x * 2) + blockstart;
        int globidx2 = globidx1 + blockDim.x;

        int sharedidx1 = threadIdx.x + threadIdx.y * blockDim.x * 2;
        int sharedidx2 = sharedidx1 + blockDim.x;
        shared_memory[sharedidx1] = polynomial_in[globidx1];
        shared_memory[sharedidx2] = polynomial_in[globidx2];

        int t_ = lg2fftsize - 1;
        int t = 1 << t_;
        
        for (int i = 0; i < lg2fftsize; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x + threadIdx.y * blockDim.x * 2;
            int idx2 = idx1 + t;

            CooleyTukeyUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t >> 1;
            t_ -= 1;
        }

        __syncthreads();
        polynomial_in[globidx1] = shared_memory[sharedidx1];
        polynomial_in[globidx2] = shared_memory[sharedidx2];
        
        return;
    }

    template <typename T>
    __global__ void ForwardCore2(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Modulus<T> mod) {
        extern __shared__ char shared_memory_typed[];
        
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x] = root_of_unity_table[threadIdx.x];

        int globIdx1 = threadIdx.x + blockIdx.x * blockDim.x * 2;
        int globIdx2 = globIdx1 + blockDim.x;

        int sharedIdx1 = threadIdx.x;
        int sharedIdx2 = threadIdx.x + blockDim.x;
        shared_memory[sharedIdx1] = polynomial_in[globIdx1];
        shared_memory[sharedIdx2] = polynomial_in[globIdx2];
        
        int t_ = LG2NUMSPERBLOCK - 1;
        int t = 1 << t_;
        
        for (int i = 0; i < LG2NUMSPERBLOCK; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x;
            int idx2 = idx1 + t;

            CooleyTukeyUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t >> 1;
            t_ -= 1;
        }

        __syncthreads();
        polynomial_in[globIdx1] = shared_memory[sharedIdx1];
        polynomial_in[globIdx2] = shared_memory[sharedIdx2];
        
        return;
    }

    template <typename T>
    __global__ void InverseCore(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Root<T> root, Modulus<T> mod, int glob_stride, int block_stride, int bitreverse_length) {
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);
        
        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x] = root_of_unity_table[threadIdx.x];
        
        int blockStart = blockIdx.y + blockIdx.x * block_stride;
        int globidx1 = blockStart + threadIdx.x * glob_stride;
        int globidx2 = globidx1 + blockDim.x * glob_stride;

        int sharedidx1 = threadIdx.x;
        int sharedidx2 = threadIdx.x + blockDim.x * blockDim.y;
        shared_memory[sharedidx1] = polynomial_in[globidx1];
        shared_memory[sharedidx2] = polynomial_in[globidx2];

        int t_ = 0;
        int t = 1 << t_;
        
        for (int i = 0; i < LG2NUMSPERBLOCK; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x;
            int idx2 = idx1 + t;

            GentlemanSandeUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t << 1;
            t_ += 1;
        }

        int brthing = bit_reverse(blockIdx.x, bitreverse_length);
        T twiddle1 = power_mod(root, brthing * sharedidx1, mod);
        T twiddle2 = power_mod(root, brthing * sharedidx2, mod);

        __syncthreads();

        shared_memory[sharedidx1] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx1], twiddle1, mod);
        shared_memory[sharedidx2] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx2], twiddle2, mod);
        polynomial_in[globidx1] = shared_memory[sharedidx1];
        polynomial_in[globidx2] = shared_memory[sharedidx2];

        return;
    }

    template <typename T>
    __global__ void InverseCore1(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Root<T> root, Modulus<T> mod, T n_inverse, int lg2fftsize, 
                                int block_stride, int glob_stride, bool scale_output) {
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);
        
        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x + threadIdx.y * blockDim.x] = root_of_unity_table[threadIdx.x + threadIdx.y * blockDim.x];

        int blockstart = blockIdx.x * block_stride;
        int globidx1 = (threadIdx.y + threadIdx.x * glob_stride) + blockstart;
        int globidx2 = globidx1 + blockDim.x * glob_stride;
        int sharedidx1 = threadIdx.x + threadIdx.y * blockDim.x * 2;
        int sharedidx2 = sharedidx1 + blockDim.x;
        shared_memory[sharedidx1] = polynomial_in[globidx1];
        shared_memory[sharedidx2] = polynomial_in[globidx2];

        int t_ = 0;
        int t = 1 << t_;
        
        for (int i = 0; i < lg2fftsize; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x + threadIdx.y * blockDim.x * 2;
            int idx2 = idx1 + t;

            GentlemanSandeUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t << 1;
            t_ += 1;
        }

        __syncthreads();
        if (scale_output) {
            shared_memory[sharedidx1] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx1], n_inverse, mod);
            shared_memory[sharedidx2] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx2], n_inverse, mod);
        }

        polynomial_in[globidx1] = shared_memory[sharedidx1];
        polynomial_in[globidx2] = shared_memory[sharedidx2];
        
        return;
    }

    template <typename T>
    __global__ void InverseCore2(T* polynomial_in, 
                                const Root<T>* __restrict__ root_of_unity_table,
                                Modulus<T> mod, T n_inverse, int global_stride, bool scale_output) {
        extern __shared__ char shared_memory_typed[];
        T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);
        
        T* small_root_table = &shared_memory[NUMSPERBLOCK];
        small_root_table[threadIdx.x] = root_of_unity_table[threadIdx.x];

        int blockstart = blockIdx.x;
        int globidx1 = threadIdx.x * global_stride + blockstart;
        int globidx2 = globidx1 + blockDim.x * global_stride;
        int sharedidx1 = threadIdx.x;
        int sharedidx2 = sharedidx1 + blockDim.x;
        shared_memory[sharedidx1] = polynomial_in[globidx1];
        shared_memory[sharedidx2] = polynomial_in[globidx2];

        int t_ = 0;
        int t = 1 << t_;
        
        for (int i = 0; i < LG2NUMSPERBLOCK; ++i) {
            __syncthreads();
            int idx1 = ((threadIdx.x >> t_) << t_) + threadIdx.x;
            int idx2 = idx1 + t;

            GentlemanSandeUnit(shared_memory[idx1], shared_memory[idx2], small_root_table[threadIdx.x >> t_], mod);

            t = t << 1;
            t_ += 1;
        }

        __syncthreads();
        if (scale_output) {
            shared_memory[sharedidx1] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx1], n_inverse, mod);
            shared_memory[sharedidx2] = OPERATOR_GPU<T>::mult(shared_memory[sharedidx2], n_inverse, mod);
        }
        
        polynomial_in[globidx1] = shared_memory[sharedidx1];
        polynomial_in[globidx2] = shared_memory[sharedidx2];
        
        return;
    }

    template <typename T>
    __host__ void GPU_CT_NTT_Inplace(T* device_inout, Root<T> root,
                                  Modulus<T> modulus, nttct_configuration<T> cfg) {
        int n = 1 << cfg.n_power;
        int lg2n = cfg.n_power;
        int block_stride, glob_stride;
        int griddim_x, griddim_y;
        int blockdim_x, blockdim_y;    

        assert(cfg.n_power >= LG2NUMSPERBLOCK);

        switch (cfg.ntt_type)
        {
            case FORWARD: {
                block_stride = n;
                glob_stride = n >> LG2NUMSPERBLOCK;
                griddim_x = n >> LG2NUMSPERBLOCK;
                griddim_y = 1;

                blockdim_x = 1 << LG2THREADSPERBLOCK;
                while (lg2n > LG2NUMSPERBLOCK) {                    
                    ForwardCore<<<dim3(griddim_x, griddim_y), blockdim_x, cfg.shared_memory>>>(device_inout, cfg.root_table, root, modulus, glob_stride, block_stride);

                    root = modular_operation_cpu::BarrettOperations<T>::exp(root, NUMSPERBLOCK, modulus);
                    lg2n -= LG2NUMSPERBLOCK;
                    block_stride >>= LG2NUMSPERBLOCK;
                    glob_stride >>= LG2NUMSPERBLOCK;
                    griddim_x >>= LG2NUMSPERBLOCK;
                    griddim_y <<= LG2NUMSPERBLOCK;
                }
                griddim_x = n >> LG2NUMSPERBLOCK;
                if (lg2n < LG2NUMSPERBLOCK) {
                    blockdim_x = 1 << (lg2n - 1);
                    blockdim_y = 1 << (LG2THREADSPERBLOCK - lg2n + 1);
                    
                    ForwardCore1<<<griddim_x, dim3(blockdim_x, blockdim_y), cfg.shared_memory>>>(device_inout, cfg.root_table, root, modulus, lg2n);
                } else /* lg2n == LG2NUMSPERBLOCK */ { 
                    blockdim_x = 1 << LG2THREADSPERBLOCK;

                    ForwardCore2<<<griddim_x, blockdim_x, cfg.shared_memory>>>(device_inout, cfg.root_table, modulus);
                }

                return;
            }
            case INVERSE: {
                block_stride = 1 << LG2NUMSPERBLOCK;
                glob_stride = 1;
                griddim_x = n >> LG2NUMSPERBLOCK;
                griddim_y = 1;

                blockdim_x = 1 << LG2THREADSPERBLOCK;

                T n_inverse;
                if (cfg.scale_output) { n_inverse = modular_operation_cpu::BarrettOperations<T>::modinv(1 << cfg.n_power, modulus); };
                
                while (lg2n > LG2NUMSPERBLOCK) {
                    lg2n -= LG2NUMSPERBLOCK;

                    InverseCore<<<dim3(griddim_x, griddim_y), blockdim_x, cfg.shared_memory>>>(device_inout, cfg.root_table, root, modulus, glob_stride, block_stride, lg2n);

                    root = modular_operation_cpu::BarrettOperations<T>::exp(root, NUMSPERBLOCK, modulus);

                    block_stride <<= LG2NUMSPERBLOCK;
                    glob_stride <<= LG2NUMSPERBLOCK;
                    griddim_x >>= LG2NUMSPERBLOCK;
                    griddim_y <<= LG2NUMSPERBLOCK;
                }
                griddim_x = n >> LG2NUMSPERBLOCK;
                if (lg2n < LG2NUMSPERBLOCK) {
                    blockdim_x = 1 << (lg2n - 1);
                    blockdim_y = 1 << (LG2THREADSPERBLOCK - lg2n + 1);

                    InverseCore1<<<griddim_x, dim3(blockdim_x, blockdim_y), cfg.shared_memory>>>(device_inout, cfg.root_table, root, modulus, n_inverse, lg2n, blockdim_y, glob_stride, cfg.scale_output);
                } else /* lg2n == LG2NUMSPERBLOCK */ {
                    blockdim_x = 1 << LG2THREADSPERBLOCK;
                    InverseCore2<<<griddim_x, blockdim_x, cfg.shared_memory>>>(device_inout, cfg.root_table, modulus, n_inverse, glob_stride, cfg.scale_output);
                }
                return;
            }
            default:
                throw std::invalid_argument("Invalid ntt_type!");
        }
    }

    template __host__ void GPU_CT_NTT_Inplace(Data64* device_inout,  Root<Data64> root,
                                  Modulus<Data64> modulus, nttct_configuration<Data64> cfg);

}