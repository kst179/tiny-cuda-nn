#include <mma.h>
#include <cstdint>
#include <chrono>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/networks/fully_fused_qat_mlp.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

using namespace tcnn;

template <typename T>
__global__ void fill_rand(uint32_t size, T* out, tcnn::default_rng_t rng) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i >= size) {
        return;
    }

    rng.advance(i);

    if (std::is_same<T, __half>::value) {
        out[i] = (T)((rng.next_float() - 0.5) * 2);
    } else {
        out[i] = (T)rng.next_uint();
    }
}

template <typename T, typename T_ACC>
__global__ void check(uint32_t size, T_ACC* expected, T* actual);

template <>
__global__ void check(uint32_t size, int32_t* expected, int8_t* actual) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    int x = expected[i];
    int y = actual[i];

    // x = max(x, 0);
    // x = (int)round((float)x / 256) - 128;
    // x = min(x, 127);

    if (x != y) {
        printf("[%d]: %d != %d\n", i, x, y);
        assert(false);
    }
}

template <>
__global__ void check(uint32_t size, __half* expected, __half* actual) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    float x = expected[i];
    float y = actual[i];

    // x = max(x, 0.0f);

    if (abs(x - y) > 5e-2) {
        printf("[%d]: %f != %f\n", i, x, y);
        assert(false);
    }
}

template <
    typename CalculationPolicy, 
    typename T_ST = typename CalculationPolicy::storage_type, 
    typename T_ACC = typename CalculationPolicy::accumulator_type
>
__global__ void test_kernel(
    const T_ST* __restrict__ input, 
    const T_ST* __restrict__ weights, 
    T_ST* __restrict__ output,
    int n_layers = 1
) {
    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    
    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t chunk_offset = chunk_idx * _::storage_chunk_size;

    extern __shared__ uint8_t shmem[];
    T_ST* act_shmem = (T_ST*)shmem;
    T_ACC* aux_shmem = (T_ACC*)((T_ST*)shmem + _::shmem_size);

    threadblock_load_input_static<CalculationPolicy>(act_shmem, input + chunk_offset);

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     for (int i = 0; i < _::chunk_height; ++i) {
    //         printf("%4d | ", i);
    //         for (int j = 0; j < WIDTH; ++j) {
    //             float x = act_shmem[i * _::shmem_stride + j];
    //             printf("%8.3f ", x);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // __syncthreads();

    #pragma unroll
    for (int i = 0; i < n_layers; ++i) {
        qat_threadblock_layer<CalculationPolicy, false, T, T_ST, T_ACC>(act_shmem, weights, nullptr, nullptr, aux_shmem);
    }

    threadblock_store_output_static<CalculationPolicy>(output + chunk_offset, act_shmem);
}

template <typename T> struct cutlass_types;
template<> struct cutlass_types< uint1b_t > { typedef cutlass::uint1b_t type; typedef int32_t acc_type; };
template<> struct cutlass_types<  int4b_t > { typedef  cutlass::int4b_t type; typedef int32_t acc_type; };
template<> struct cutlass_types< uint4b_t > { typedef cutlass::uint1b_t type; typedef int32_t acc_type; };
template<> struct cutlass_types<   int8_t > { typedef            int8_t type; typedef int32_t acc_type; };
template<> struct cutlass_types<  uint8_t > { typedef           uint8_t type; typedef int32_t acc_type; };
template<> struct cutlass_types<   __half > { typedef   cutlass::half_t type; typedef  __half acc_type; };

// template<typename CalculationPolicy>

template <typename CalculationPolicy>
void test(uint32_t batch_size) {
    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    using T_ST = typename CalculationPolicy::storage_type;
    using T_ACC = typename CalculationPolicy::accumulator_type;

    using T_CUTL = typename cutlass_types<T>::type;
    using T_OUT_CUTL = typename cutlass_types<T>::acc_type;

    using GemmOp = cutlass::gemm::device::Gemm<
        T_CUTL,                                 // A type
        cutlass::layout::RowMajor,              // A layout
        T_CUTL,                                 // B type
        cutlass::layout::ColumnMajor,           // B layout
        T_OUT_CUTL,                             // Out type
        cutlass::layout::RowMajor,              // Out layout
        T_OUT_CUTL                              // Accumulator type
    >;

    GPUMatrix<T_ST, MatrixLayout::RowMajor> input(batch_size, _::storage_width);
    GPUMatrix<T_ST, MatrixLayout::ColumnMajor> weights(_::storage_width, _::chunk_width);
    GPUMatrix<T_ACC, MatrixLayout::RowMajor> output_expected(batch_size, _::chunk_width);
    GPUMatrix<T_ST, MatrixLayout::RowMajor> output_actual(batch_size, _::storage_width);

    default_rng_t rng{1337};

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, input.n_elements(), input.data(), rng);

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, weights.n_elements(), weights.data(), rng);

    GemmOp gemm_op;
    cutlass::Status status;
    
    status = gemm_op({
        {(int)batch_size, (int)_::chunk_width, (int)_::chunk_width},
        {(T_CUTL*)input.data(), (int)input.stride()},
        {(T_CUTL*)weights.data(), (int)weights.stride()},
        {(T_OUT_CUTL*)output_expected.data(), (int)output_expected.stride()},
        {(T_OUT_CUTL*)output_expected.data(), (int)output_expected.stride()},
        {1, 0}
    });

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("Got cutlass error: ") + cutlass::cutlassGetStatusString(status));
    }

    assert(batch_size % _::chunk_height == 0);

    dim3 threads { 32, _::n_warps, 1 };
    dim3 blocks { batch_size / _::chunk_height, 1, 1 };

    std::cout << "storage_size: " << sizeof(T_ST) << "\n"
              << "accum_size: " << sizeof(T_ACC) << "\n"
              << "block_height: " <<  _::block_height << "\n"
              << "block_width: " << _::block_width << "\n"
              << "n_blocks_col: " << _::n_blocks_col << "\n" 
              << "n_blocks_row: " <<  _::n_blocks_row << "\n"
              << "n_weight_blocks: " <<  _::n_weight_blocks << "\n"
              << "n_warps_col: " << _::n_warps_col << "\n"
              << "n_warps_row: " << _::n_warps_row << "\n"
              << "n_iters_col: " << _::n_iters_col << "\n"
              << "n_iters_row: " << _::n_iters_row << "\n"
              << "width_offset: " << _::width_offset << "\n"
              << "height_offset: " << _::height_offset << "\n"
              << "chunk_height: " << _::chunk_height << "\n"
              << "chunk_size: " << _::chunk_size << "\n" 
              << "storage_chunk_size: " << _::storage_chunk_size << "\n" 
              << "storage_width: " <<_::storage_width << "\n"
              << "storage_block_width: " <<_::storage_block_width << "\n"
              << "shmem_skew: " <<_::shmem_skew << "\n"
              << "shmem_stride: " <<_::shmem_stride << "\n"
              << "shmem_size: " <<_::shmem_size << "\n"
              << "acc_shmem_skew: " <<_::acc_shmem_skew << "\n"
              << "acc_shmem_stride: " <<_::acc_shmem_stride << "\n"
              << "acc_shmem_size: " <<_::acc_shmem_size << "\n"
              << "num_lanes: " << threads.x << "\n"
              << "num_warps: " << threads.y << "\n"
              << "num_blocks: " << blocks.x << "\n"
              << std::endl;

    int shmem_size_bytes = _::shmem_size * sizeof(T_ST) + 
                           (std::is_same<T, T_ACC>::value ? 0 : _::acc_shmem_size * sizeof(T_ACC));

    std::cout << shmem_size_bytes << std::endl;
    std::cout << _::shmem_size * sizeof(T_ST) << " " << _::acc_shmem_size * sizeof(T_ACC) << std::endl;

    // int32_t scale_exp, scale_frac; 
    // decompose_float(1.0 / 256.0, &scale_exp, &scale_frac);

    CUDA_CHECK_THROW(cudaFuncSetAttribute(test_kernel<CalculationPolicy>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes));
    test_kernel<CalculationPolicy><<<blocks, threads, shmem_size_bytes>>>(input.data(), weights.data(), output_actual.data());

    // linear_kernel(check<T, T_ACC>, 0, nullptr, input.n_elements(), input.data(), output_actual.data());
    // linear_kernel(check<T, T_ACC>, 0, nullptr, output_expected.n_elements(), output_expected.data(), output_actual.data());
 
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    std::cout << "OK!" << std::endl;

#if true
    std::vector<T_ST> input_cpu(batch_size * _::storage_width);
    std::vector<T_ACC> output_expected_cpu(batch_size * _::storage_width);
    std::vector<T_ST> output_actual_cpu(batch_size * _::storage_width);

    CUDA_CHECK_THROW(cudaMemcpy(input_cpu.data(), input.data(), input.n_elements() * sizeof(T_ST), cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(output_expected_cpu.data(), output_expected.data(), output_expected.n_elements() * sizeof(T_ACC), cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(output_actual_cpu.data(), output_actual.data(), output_actual.n_elements() * sizeof(T_ST), cudaMemcpyDeviceToHost));

    printf("INPUT\n");
    for (int i = 0; i < min(batch_size, 16); ++i) {
        printf("%4d I ", i);
        for (int j = 0; j < _::storage_width; ++j) {
            if (std::is_same<T, __half>::value) {
                printf("%8.3f ", (float)input_cpu[i * _::storage_width + j]);
            } else if(std::is_same<T, int8_t>::value) {
                int x = (int8_t)(float)input_cpu[i * _::storage_width + j];
                printf("%5d ", (int)x);
            } else {
                uint32_t x = input_cpu[i * _::storage_width + j];
                for (int k = 0; k < 32; ++k) {
                    printf((x & (1 << k)) ? "1" : "0");
                    if (k % 8 == 7) printf(" ");
                }
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("\n\n");

    for (int i = 0; i < min(batch_size, 16); ++i) {
        printf("%4d E ", i);
        for (int j = 0; j < _::storage_width; ++j) {
            if (std::is_same<T, __half>::value) {
                printf("%8.3f ", (float)output_expected_cpu[i * _::storage_width + j]);
            } else if(std::is_same<T, int8_t>::value) {
                int x = (int8_t)(float)output_expected_cpu[i * _::storage_width + j];
                // x = max(x, 0);
                // x = (int)round((float)x / 256) - 128;
                // x = min(x, 127);

                printf("%5d ", (int)x);
            } else {
                uint32_t x = output_expected_cpu[i * _::storage_width + j];
                for (int k = 0; k < 32; ++k) {
                    printf((x & (1 << k)) ? "1" : "0");
                    if (k % 8 == 7) printf(" ");
                }
                printf(" ");
            }

        }
        printf("\n     A ");

        for (int j = 0; j < _::storage_width; ++j) {
            if (std::is_same<T, __half>::value) {
                printf("%8.3f ", (float)output_actual_cpu[i * _::chunk_width + j]);
            } else if(std::is_same<T, int8_t>::value) {
                printf("%5d ", (int)output_actual_cpu[i * _::chunk_width + j]);
            } else {
                uint32_t x = output_actual_cpu[i * _::storage_width + j];
                for (int k = 0; k < 32; ++k) {
                    printf((x & (1 << k)) ? "1" : "0");
                    if (k % 8 == 7) printf(" ");
                }
                printf(" ");
            }
        }
        printf("\n\n");
    }
#endif

}


template <typename CalculationPolicy>
void bench(uint32_t batch_size) {
    // constexpr bool HALF = std::is_same<T, __half>::value;
    // using T_ACC = std::conditional_t<HALF, __half, int32_t>;

    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    using T_ST = typename CalculationPolicy::storage_type;
    using T_ACC = typename CalculationPolicy::accumulator_type;

    GPUMatrix<T_ST, MatrixLayout::RowMajor> input(batch_size, _::storage_width);
    GPUMatrix<T_ST, MatrixLayout::ColumnMajor> weights(_::storage_width, _::chunk_width);
    GPUMatrix<T_ST, MatrixLayout::RowMajor> output(batch_size, _::storage_width);

    default_rng_t rng{1337};

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, input.n_elements(), input.data(), rng);

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, weights.n_elements(), weights.data(), rng);

    assert(batch_size % _::chunk_height == 0);

    dim3 threads { 32, _::n_warps, 1 };
    dim3 blocks { batch_size / _::chunk_height, 1, 1 };

    // int32_t scale_exp, scale_frac;
    // decompose_float(1.0 / 256.0, &scale_exp, &scale_frac);

    int shmem_size_bytes = _::shmem_size * sizeof(T_ST) + 
                           (std::is_same<T, T_ACC>::value ? 0 : _::acc_shmem_size * sizeof(T_ACC));

    CUDA_CHECK_THROW(cudaFuncSetAttribute(test_kernel<CalculationPolicy>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    
    constexpr int n_batches = 100;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_batches; ++i) {
        test_kernel<CalculationPolicy><<<blocks, threads, shmem_size_bytes>>>(input.data(), weights.data(), output.data(), 8);
    }

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    std::cout << std::setw(10) << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1'000'000 / n_batches << " ms/batch" << std::endl;
}


template<typename T, typename U = storage_t<T>>
void func() {
    std::cout << sizeof(U) << std::endl;
}

int main () {
    // func<__half>();
    // func<bit4_t>();
    // tcnn::GPUMatrix<bit_t> a(128, 128);
    // std::cout << sizeof(a.data()) << std::endl;
    // std::cout << sizeof(typename bit_t::storage_element_type) << std::endl;

    using HalfTestPolicy  = FulllyFusedMatMulCalculationPolicy<   __half, /*WIDTH=*/128, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 8, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;
    using Int8TestPolicy  = FulllyFusedMatMulCalculationPolicy<   int8_t, /*WIDTH=*/128, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 8, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;
    using BMMTestPolicy4  = FulllyFusedMatMulCalculationPolicy< uint1b_t, /*WIDTH=*/128, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 4, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;
    using BMMTestPolicy8  = FulllyFusedMatMulCalculationPolicy< uint1b_t, /*WIDTH=*/128, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 8, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint2    >;
    using BMMTestPolicy16 = FulllyFusedMatMulCalculationPolicy< uint1b_t, /*WIDTH=*/128, /*HEIGHT=*/128, /*N_WARPS_COL=*/16, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint32_t >;
    // using BMMTestPolicy16 = FulllyFusedMatMulCalculationPolicy< uint1b_t, /*WIDTH=*/256, /*HEIGHT=*/128, /*N_WARPS_COL=*/16, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint32_t >;


    using HalfTestPolicy64  = FulllyFusedMatMulCalculationPolicy<   __half, /*WIDTH=*/64, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 4, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;
    using Int8TestPolicy64  = FulllyFusedMatMulCalculationPolicy<   int8_t, /*WIDTH=*/64, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 4, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;
    using Int4TestPolicy64  = FulllyFusedMatMulCalculationPolicy<  int4b_t, /*WIDTH=*/64, /*HEIGHT=*/128, /*N_WARPS_COL=*/ 8, /*N_WARPS_ROW=*/1, /*T_COPY=*/uint4    >;

    // test<HalfTestPolicy>(1024);
    // test<Int8TestPolicy>(1024);
    // test<BMMTestPolicy>(1024);
    // test<int8_t>();
    // test<__half>();

    // size_t free, total;
    // cudaMemGetInfo(&free, &total);

    // uint32_t weights_size = 64 * 64 * sizeof(__half);
    // uint32_t sample_size = 64 * sizeof(__half);

    size_t batch_size = 1 << 23; //  (free - weights_size) * 0.95 / (2 * sample_size);

    // bench<HalfTestPolicy>(batch_size);
    // bench<Int8TestPolicy>(batch_size);
    // bench<BMMTestPolicy4>(batch_size);
    // bench<BMMTestPolicy8>(batch_size);
    // bench<BMMTestPolicy16>(batch_size);

    bench<HalfTestPolicy64>(batch_size);
    bench<Int8TestPolicy64>(batch_size);
    bench<Int4TestPolicy64>(batch_size);

    return 0;
}