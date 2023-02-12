/**
 * @copyright Copyright (c) 2023
 *
 * @file fully_fused_qat_mlp.h
 * @date 2023-02-04
 * @author Kozlovtsev Konstantin (kozlovtsev179@gmail.com)
 * @brief Implementation of fully fused network adapted to quantization aware training.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/type_traits.h>

#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T, uint32_t WIDTH, uint32_t HEIGHT, uint32_t N_WARPS_COL, uint32_t N_WARPS_ROW, typename COPY_T=uint4>
struct FulllyFusedMatMulCalculationPolicy {
private:
    template <bool ACC>
    static constexpr uint32_t define_skew() {
        constexpr uint32_t block_width = helper_traits<T>::block_width;
        constexpr uint32_t storage_block_width = ACC ? block_width : block_width / helper_traits<T>::elements_per_storage_unit;
        constexpr uint32_t storage_size = ACC ? sizeof(accumulator_t<T>) : sizeof(storage_t<T>);

        // 128 = 4 * 32 the memory (in bytes) which can be acessed by all shared banks at once
        // if block_width == WIDTH then there is no unused banks while copying one block
        if (storage_block_width * storage_size >= 128 || block_width == WIDTH) {
            return 0;
        } else {
            return 16 / storage_size;
        }
    }

public:
    typedef T mm_type;
    typedef storage_t<T> storage_type;
    typedef accumulator_t<T> accumulator_type;
    typedef COPY_T copy_type;

    static const uint32_t chunk_width           = WIDTH;
    static const uint32_t chunk_height          = HEIGHT;
    static const uint32_t n_warps_col           = N_WARPS_COL;
    static const uint32_t n_warps_row           = N_WARPS_ROW;

    static const uint32_t n_warps               = n_warps_row * n_warps_col;
    static const uint32_t warp_size             = 32;
    
    static const uint32_t block_height          = helper_traits<T>::block_height;
    static const uint32_t block_width           = helper_traits<T>::block_width;

    static const uint32_t n_blocks_col          = chunk_width / block_height;
    static const uint32_t n_blocks_row          = chunk_height / block_height;
    static const uint32_t n_weight_blocks       = chunk_width / block_width;

    static const uint32_t n_iters_col           = n_blocks_col / n_warps_col;
    static const uint32_t n_iters_row           = n_blocks_row / n_warps_row;

    static const uint32_t chunk_size            = chunk_height * chunk_width;

    static const uint32_t width_offset          = n_iters_col * block_height;
    static const uint32_t height_offset         = n_iters_row * block_height;

    static const uint32_t storage_width         = WIDTH / helper_traits<T>::elements_per_storage_unit;
    static const uint32_t storage_block_width   = block_width / helper_traits<T>::elements_per_storage_unit;
    static const uint32_t storage_chunk_size    = chunk_height * storage_width;
    static const uint32_t storage_ldm           = chunk_width;

    static const uint32_t shmem_skew            = define_skew<false>();
    static const uint32_t shmem_stride          = storage_width + shmem_skew;
    static const uint32_t shmem_size            = chunk_height * shmem_stride;
    static const uint32_t shmem_ldm             = shmem_stride * helper_traits<T>::elements_per_storage_unit;

    static const uint32_t acc_shmem_skew        = define_skew<true>();
    static const uint32_t acc_shmem_stride      = WIDTH + acc_shmem_skew;
    static const uint32_t acc_shmem_size        = chunk_height * acc_shmem_stride;
    static const uint32_t acc_shmem_ldm         = acc_shmem_stride;

    static const uint32_t copy_stride = sizeof(copy_type) / sizeof(storage_type);
    static const uint32_t copy_step = n_warps * warp_size * copy_stride;

    // static checks
    static_assert(chunk_width % n_warps_col == 0, "chunk_width should be divisible by n_warps_col");
    static_assert(chunk_height % n_warps_row == 0, "chunk_height should be divisible by n_warps_row");
    static_assert(sizeof(copy_type) >= sizeof(storage_type), "Size of copy type should not be less than size of storage type");

    static_assert(storage_chunk_size % copy_step == 0, "Chunk size should be divisible by the amount of memory which can be loaded or stored by one threadblock. "
                                                       "This is needed to make loads/stores in integer number of threadblock-wise copies (to support SIMT execution). "
                                                       "Try increase HEIGHT, decrease number of warps, or select smaller T_COPY, to fix this.");
    static_assert(copy_step % storage_width == 0, "");
};

// constexpr int32_t MANTISSA_LEN_BITS = 2;

// /*
//  * Finds exponent(E) and fraction(M) parts of a float/double value in range (0, 1) represented as 
//  *     x = M * 2^(-E - 32)
//  */
// __host__ __device__ inline void decompose_float(double value, int32_t* exponent, int32_t* fraction) {
//     assert(0.0 <= value && value <= 1.0);
//     *exponent = floor(-log2(value));
//     *fraction = round(value * (double)(1 << (*exponent + MANTISSA_LEN_BITS)));
// }

// __device__ inline int32_t muliply_and_round(int32_t x,  int32_t exponent, int32_t fraction) {
//     return ((uint64_t)x * (uint64_t)fraction + (1ull << (MANTISSA_LEN_BITS + exponent - 1)) - 1) >> (exponent + MANTISSA_LEN_BITS);
// }


/**
 * Copies chunk from input_threadblock to act_shmem, assuming act_shmem contains 16-byte skew
 */ 
template <
    typename CalculationPolicy,
    typename T_ST = typename CalculationPolicy::storage_type,
    typename T_COPY = typename CalculationPolicy::copy_type
>
__device__ void threadblock_load_input_static(
          T_ST* __restrict__ act_shmem, 
    const T_ST* __restrict__ input_threadblock
) {
    using _ = CalculationPolicy;

	const uint32_t lane_idx = threadIdx.x;
	const uint32_t warp_idx = threadIdx.y;

	const uint32_t col = (lane_idx * _::copy_stride) % _::storage_width;
	const uint32_t row = (lane_idx + warp_idx * _::warp_size) * _::copy_stride / _::storage_width;

    // each lane (of WARP_SIZE lanes) copies COPY_STRIDE elements in single vector uint4 operation (or different T_COPY type, if specified)
    // each warp (of N_WARPS warps) copies WARP_SIZE * COPY_STRIDE elements
    // full threadblock copies then N_WARPS * WARP_SIZE * COPY_STRIDE elements at once, 
    // then repeats untill all of CHUNK_SIZE elements are copied.
	TCNN_PRAGMA_UNROLL
	for (uint32_t offset = 0; offset < _::storage_chunk_size; offset += _::copy_step) {
        const uint32_t shmem_offset = offset / _::storage_width * _::shmem_stride;

		*(T_COPY*)&act_shmem[shmem_offset + row * _::shmem_stride + col] = 
            *(T_COPY*)&input_threadblock[offset + row * _::storage_width + col];
	}

    __syncthreads();
}


/*
 * Copies chunk from act_shmem to output_threadblock, assuming act_shmem contains 16-byte skew
 */ 
template <
    typename CalculationPolicy, 
    typename T_ST = typename CalculationPolicy::storage_type, 
    typename T_COPY = typename CalculationPolicy::copy_type
>
__device__ void threadblock_store_output_static(
          T_ST* __restrict__ output_threadblock, 
    const T_ST* __restrict__ act_shmem
) {
    using _ = CalculationPolicy;

	const uint32_t lane_idx = threadIdx.x;
	const uint32_t warp_idx = threadIdx.y;

	const uint32_t col = (lane_idx * _::copy_stride) % _::storage_width;
	const uint32_t row = (lane_idx + warp_idx * _::warp_size) * _::copy_stride / _::storage_width;

	TCNN_PRAGMA_UNROLL
	for (int offset = 0; offset < _::storage_chunk_size; offset += _::copy_step) {
        const uint32_t shmem_offset = offset / _::storage_width * _::shmem_stride;

		*(T_COPY*)&output_threadblock[offset + row * _::storage_width + col] = *(T_COPY*)&act_shmem[shmem_offset + row * _::shmem_stride + col];
	}

    __syncthreads();
}

template <typename T, typename CalculationPolicy>
struct NumericConverter;

template <typename CalculationPolicy>
struct NumericConverter<__half, CalculationPolicy> { 
    TCNN_DEVICE 
    static inline void convert(storage_t<__half>* __restrict__ act_shmem, const accumulator_t<__half>* __restrict__ aux_shmem) { 
        assert(false); // Not implemented error
    }
};

template <typename CalculationPolicy>
struct NumericConverter<int8_t, CalculationPolicy> {
    TCNN_DEVICE
    static inline void convert(storage_t<int8_t>* __restrict__ act_shmem, const accumulator_t<int8_t>* __restrict__ aux_shmem) { 
        using _ = CalculationPolicy;

        // sizeof(int4)    == 4 * sizeof(int32_t)
        // sizeof(int32_t) == 4 * sizeof(int8_t)
        constexpr uint32_t copy_stride = 4;
        constexpr uint32_t step = _::n_warps * _::warp_size * copy_stride;

        static_assert(_::chunk_size % step == 0);
        static_assert(step % _::storage_width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t col = (lane_idx * copy_stride) % _::storage_width;
        const uint32_t row = (lane_idx + warp_idx * _::warp_size) * copy_stride / _::storage_width;

        int32_t int32x4[4];
        int8_t int8x4[4];

        TCNN_PRAGMA_UNROLL
        for (uint32_t offset = 0; offset < _::chunk_size; offset += step) {
            const uint32_t aux_shmem_offset = offset / _::chunk_width * _::acc_shmem_stride;
            const uint32_t shmem_offset = offset / _::chunk_width * _::shmem_stride;

            // load 4 x int32_t at once
            *(uint4*)int32x4 = *(uint4*)&aux_shmem[aux_shmem_offset + row * _::acc_shmem_stride + col]; 

            // probably bank conflicts in uint4 copy will break the SIMT model so we fix it here (but only for single warp)
            __syncwarp();
            
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 4; ++i) {
                int8x4[i] = (uint8_t)int32x4[i];
            }

            // store 4 x int8_t at once
            *(uint32_t*)&act_shmem[shmem_offset + row * _::shmem_stride + col] = *(uint32_t*)int8x4;
        }

        __syncthreads();
    }
};

template <typename CalculationPolicy>
struct NumericConverter<int4b_t, CalculationPolicy> {
    TCNN_DEVICE 
    static inline void convert(storage_t<int4b_t>* __restrict__ act_shmem, const accumulator_t<int4b_t>* __restrict__ aux_shmem) { 
        using _ = CalculationPolicy;
        
        constexpr uint32_t copy_stride = 8;
        constexpr uint32_t step = _::n_warps * _::warp_size * copy_stride;

        static_assert(_::chunk_size % step == 0);
        static_assert(step % _::chunk_width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t col = (lane_idx * copy_stride) % _::chunk_width;
        const uint32_t row = (lane_idx + warp_idx * _::warp_size) * copy_stride / _::chunk_width;

        const uint32_t storage_col = col / helper_traits<int4b_t>::elements_per_storage_unit;

        TCNN_PRAGMA_UNROLL
        for (uint32_t offset = 0; offset < _::chunk_size; offset += step) {
            const uint32_t aux_shmem_offset = offset / _::chunk_width * _::acc_shmem_stride;
            const uint32_t shmem_offset = offset / _::chunk_width * _::shmem_stride;

            uint32_t int4bx8 = 0;

            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t j = ((lane_idx / 4 + i) % 2) * 4;
                const uint4 vec = *(uint4*)&aux_shmem[aux_shmem_offset + row * _::acc_shmem_stride + col + j];

                int4bx8 |= (vec.x & 0xF) << (4 * j +  0);
                int4bx8 |= (vec.y & 0xF) << (4 * j +  4);
                int4bx8 |= (vec.z & 0xF) << (4 * j +  8);
                int4bx8 |= (vec.w & 0xF) << (4 * j + 12);
            }

            __syncwarp();

            act_shmem[shmem_offset + row * _::shmem_stride + storage_col] = int4bx8;
        }

        __syncthreads();

    }
};

template <typename CalculationPolicy>
struct NumericConverter<uint1b_t, CalculationPolicy> {
    TCNN_DEVICE 
    static inline void convert(storage_t<uint1b_t>* __restrict__ act_shmem, const accumulator_t<uint1b_t>* __restrict__ aux_shmem) { 
        using _ = CalculationPolicy;
        
        constexpr uint32_t copy_stride = 32;
        constexpr uint32_t step = _::n_warps * _::warp_size * copy_stride;

        static_assert(_::chunk_size % step == 0);
        static_assert(step % _::chunk_width == 0);

        const uint32_t lane_idx = threadIdx.x;
        const uint32_t warp_idx = threadIdx.y;

        const uint32_t col = (lane_idx * copy_stride) % _::chunk_width;
        const uint32_t row = (lane_idx + warp_idx * _::warp_size) * copy_stride / _::chunk_width;

        const uint32_t storage_col = col / helper_traits<uint1b_t>::elements_per_storage_unit;

#if false
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < min(_::chunk_height, 16); ++i) {
                printf("%4d | ", i);
                for (int j = 0; j < _::chunk_width; ++j) {
                    uint32_t x = aux_shmem[i * _::acc_shmem_stride + j];
                    printf((x & 1) ? "1" : "0");
                    if (j % 8 == 7) printf(" ");
                    if (j % 32 == 31) printf(" ");
                }
                printf("\n");
            }
            printf("\n\n");
        }
        __syncthreads();
#endif

        TCNN_PRAGMA_UNROLL
        for (uint32_t offset = 0; offset < _::chunk_size; offset += step) {
            const uint32_t aux_shmem_offset = offset / _::chunk_width * _::acc_shmem_stride;
            const uint32_t shmem_offset = offset / _::chunk_width * _::shmem_stride;

            uint32_t bits32 = 0;

            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t j = ((lane_idx + i) % 8) * 4;
                const uint4 vec = *(uint4*)&aux_shmem[aux_shmem_offset + row * _::acc_shmem_stride + col + j];

                bits32 |= (vec.x & 1) << (j + 0);
                bits32 |= (vec.y & 1) << (j + 1);
                bits32 |= (vec.z & 1) << (j + 2);
                bits32 |= (vec.w & 1) << (j + 3);
            }

            __syncwarp();

            act_shmem[shmem_offset + row * _::shmem_stride + storage_col] = bits32;
        }

        __syncthreads();

#if false
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            for (int i = 0; i < min(_::chunk_height, 16); ++i) {
                printf("%4d | ", i);
                for (int j = 0; j < _::storage_width; ++j) {
                    uint32_t x = act_shmem[i * _::shmem_stride + j];
                    for (int k = 0; k < 32; ++k) {
                        printf((x & (1 << k)) ? "1" : "0");
                        if (k % 8 == 7) printf(" ");
                    }
                    printf(" ");
                }
                printf("\n");
            }
            printf("\n\n");
        }

        __syncthreads();
#endif

    }
};

/*
template<uint32_t WIDTH, uint32_t N_ITERS, uint32_t N_WARPS>
__device__ inline void downcast_copy(__half* __restrict__ act_shmem, const __half* __restrict__ aux_shmem) {
    assert(false); // Not implemented error
}

template<uint32_t WIDTH, uint32_t N_ITERS, uint32_t N_WARPS>
__device__ inline void downcast_copy(int8_t* __restrict__ act_shmem, const int32_t* __restrict__ aux_shmem) { 
                                                //   const int32_t scale_exp, const int32_t scale_frac, const int32_t* quantization_shift) {
    using _ = CalculationPolicy<int8_t, WIDTH, N_ITERS, N_WARPS>;

    // DEFINE_COMMON_CONSTEXPR
    // constexpr uint32_t AUX_SHM_SKEW = 16 / sizeof(T_ACC);
    // constexpr uint32_t AUX_SHM_STRIDE = WIDTH + AUX_SHM_SKEW;

    // sizeof(int4)    == 4 * sizeof(int32_t) 
    // sizeof(int32_t) == 4 * sizeof(int8_t)
    constexpr uint32_t COPY_STRIDE = 4;
    constexpr uint32_t STEP = N_WARPS * WARP_SIZE * COPY_STRIDE;

    static_assert(_::storage_chunk_size % STEP == 0);
    static_assert(STEP % _::storage_width == 0);

    const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

    const uint32_t col = (li * COPY_STRIDE) % _::storage_width;
    const uint32_t row = (li + wi * WARP_SIZE) * COPY_STRIDE / _::storage_width;

    int32_t int32x4[4];
    int8_t int8x4[4];

    TCNN_PRAGMA_UNROLL
    for (uint32_t offset = 0; offset < _::storage_chunk_size; offset += STEP) {
        const uint32_t aux_shmem_offset = offset / _::storage_width * _::acc_shmem_stride;
        const uint32_t shmem_offset = offset / _::storage_width * _::shmem_stride;

        // load 4 x int32_t at once
        *(uint4*)int32x4 = *(uint4*)&aux_shmem[aux_shmem_offset + row * _::acc_shmem_stride + col]; 

        // probably bank conflicts in uint4 copy will break the SIMT model so we fix it here (but only for single warp)
        __syncwarp();

        // downcast + optional shif, rescale, relu, crop
        TCNN_PRAGMA_UNROLL
        for (uint32_t i = 0; i < 4; ++i) {
            int8x4[i] = (uint8_t)int32x4[i];

            // if there is no quantization shift, then this operations done inplace in tensor core's registers
            // if (quantization_shift) {
                // x += quantization_shift[col + i];                      // Add shift before ReLU
                // x = max(x, 0);                                         // Apply ReLU
                // x = muliply_and_round(x, scale_exp, scale_frac) + MIN_BYTE;
                // x = min(x, MAX_BYTE);                                  // Crop outliers
            // }
        }

        // store 4 x int8_t at once
        *(uint32_t*)&act_shmem[shmem_offset + row * _::shmem_stride + col] = *(uint32_t*)int8x4;
    }

    __syncthreads();
}
*/

template<typename F_A, typename F_B, typename F_ACC>
__device__ void mma_sync(F_ACC& result_frag, const F_A& act_frag, const F_B& weights_frag) {
    nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag, result_frag);
}

template<>
__device__ void mma_sync(
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int32_t>& result_frag,
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, uint1b_t, nvcuda::wmma::row_major>& act_frag, 
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, uint1b_t, nvcuda::wmma::col_major>& weights_frag) {

    nvcuda::wmma::bmma_sync(result_frag, act_frag, weights_frag, result_frag);
}

/*
 * Single-layer pass (fused matmul, activation and fake quantiztion) can be used for forward and backward pass.
 * For the last one weights are transposed, activations replaced with its derivatives and no fake quantization done.
 */
template <
    typename CalculationPolicy,
    bool BACKWARD = false,
    typename T = typename CalculationPolicy::mm_type,
    typename T_ST = typename CalculationPolicy::storage_type, 
    typename T_ACC = typename CalculationPolicy::accumulator_type
>
__device__ void qat_threadblock_layer(
    // Activation activation,
          T_ST* __restrict__ act_shmem,
    const T_ST* __restrict__ weights_this_layer,
          T_ST* __restrict__ out_intermediate_threadblock_this_layer = nullptr,
    const T_ST* __restrict__ activation_aux = nullptr,
          T_ACC* __restrict__ aux_shmem = nullptr
) {
	using namespace nvcuda;
    using _ = CalculationPolicy;

    constexpr bool half_mode = std::is_same<T, __half>::value;

    static_assert(!BACKWARD || half_mode, "Backward threadblock layer is available only for T=__half, "
                                          "integer types should be used for inference only!");

    static_assert(!half_mode || _::n_iters_col == 1, "If use half mode, matmul result is stored inplace into shared memory, "
                                                     "so we write to the matrix rows from which then read for further multiplications, "
                                                     "if all operations are not done in parallel. "
                                                     "To obtain this behavior try set N_WARPS_COLS = WIDTH / 16");
                                                     
	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

    constexpr uint32_t M = _::block_height;
    constexpr uint32_t N = _::block_height;
    constexpr uint32_t K = _::block_width;

	// Fragments
	wmma::fragment<wmma::matrix_a, M, N, K, T, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, T, weights_layout_t> weights_frag[_::n_weight_blocks];
	wmma::fragment<wmma::accumulator, M, N, K, T_ACC> result_frag[_::n_iters_row];

	const uint32_t warp_idx = threadIdx.y;

	const uint32_t col_offset = _::width_offset * (warp_idx % _::n_warps_col);
    const uint32_t row_offset = _::height_offset * (warp_idx / _::n_warps_col);

#if false
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("warp: %d, col_offset: %d, row_offset %d\n", warp_idx, col_offset, row_offset);
    // }

    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("INPUT SHMEM:\n");
        for (int i = 0; i < min(_::chunk_height, 16); ++i) {
            printf("%4d | ", i);
            for (int j = 0; j < _::chunk_width; ++j) {
                float x = act_shmem[i * _::shmem_stride + j];
                printf("%8.3f ", x);
            }
            printf("\n");
        }
        printf("\n");
    }
	__syncthreads();
#endif

#if false
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < min(_::chunk_height, 16); ++i) {
            printf("%4d | ", i);
            for (int j = 0; j < _::storage_width; ++j) {
                uint32_t x = act_shmem[i * _::shmem_stride + j];
                for (int k = 0; k < 32; ++k) {
                    printf((x & (1 << k)) ? "1" : "0");
                    if (k % 8 == 7) printf(" ");
                }
                printf(" ");
            }
            printf("\n");
        }
        printf("\n\n");
    }

    __syncthreads();
#endif

    TCNN_PRAGMA_UNROLL
    // for (uint32_t iter_col = 0; iter_col < _::n_iters_col; ++iter_col) {
    for (int32_t iter_col = _::n_iters_col-1; iter_col > -1; --iter_col) {
        const uint32_t weights_col = col_offset + iter_col * _::block_height;
     
        // Load N_BLOCKS chunks of weights from global memory into registers.
        TCNN_PRAGMA_UNROLL
        for (uint32_t block = 0; block < _::n_weight_blocks; ++block) {
            const uint32_t row = block * _::storage_block_width;

            // if (blockIdx.x == 0 && threadIdx.x == 0) printf("wi: %d, load weights at [%d, %d] to %d weight block\n", warp_idx, row, weights_col, block);

            if (BACKWARD) {
                // If we're performing the backward pass, additional index swizzling is needed to load the weights in transposed form.
                // BACKWARD is available only for T=__half so chunk_width is used as stride here.
                wmma::load_matrix_sync(weights_frag[block], weights_this_layer + row * _::chunk_width + weights_col, _::chunk_width);
            } else {
                wmma::load_matrix_sync(weights_frag[block], weights_this_layer + row + weights_col * _::storage_width, _::chunk_width);
            }
        }

        TCNN_PRAGMA_UNROLL
        for (int iter_row = 0; iter_row < _::n_iters_row; ++iter_row) {
            const uint32_t row = row_offset + iter_row * _::block_height;
            
            wmma::fill_fragment(result_frag[iter_row], 0);

            TCNN_PRAGMA_UNROLL
            for (uint32_t weight_block = 0; weight_block < _::n_weight_blocks; ++weight_block) {
                const uint32_t col = weight_block * _::storage_block_width;

                // if (blockIdx.x == 0 && threadIdx.x == 0) printf("wi: %d, load acts at [%d, %d], multiply by %d weight block & add to %d acc\n", warp_idx, row, col, weight_block, iter_row);

                // Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
                wmma::load_matrix_sync(act_frag, act_shmem + row * _::shmem_stride + col, _::shmem_ldm);
                mma_sync(result_frag[iter_row], act_frag, weights_frag[weight_block]);
            }

    /*
            // Activation
            // if (INFERENCE) {
            //     // If there is no columnwise shift before ReLU activation, we can apply activation inside the registers (supposed to be faster) 
            //     // else it is done alongside with data copy from accumulator memory to shared activations
            //     // if (quantization_shift == nullptr) {
            //         // TCNN_PRAGMA_UNROLL
            //         // for (uint32_t t = 0; t < result_frag[iter].num_elements; ++t) {
            //         //     int32_t x = result_frag[iter].x[t];

            //         //     x = max(x, 0);                                                 // ReLU activation
            //         //     x = muliply_and_round(x, scale_exp, scale_frac) + MIN_BYTE;    // scale, round and shift to [-128, +128)
            //         //     x = min(x, MAX_BYTE);                                          // crop the outliers if x > 127

            //         //     result_frag[iter].x[t] = x;
            //         // }
            //     // }
            // } else if (BACKWARD) {
            // 	// Load the temporary forward matrix for the relu transfer
            // 	wmma::load_matrix_sync(act_frag, activation_aux + row * WIDTH + weights_col, WIDTH);
            // 	warp_activation_backward<T>(Activation::ReLU, result_frag[iter], act_frag, result_frag[iter]);
            // } else {
            // 	warp_activation<T>(Activation::ReLU, result_frag[iter], result_frag[iter]);
            // }
    */
        }

        // Barrier ensures that all blocks multiplications are done before writing to act_smem (some warps can still read from it)
        // Because for non-half types we write result to auxillary shared memory, we do not need it in implementations for integer types
        if (half_mode) {
            __syncthreads();
        }

        // Copy fragments back to shared memory
        TCNN_PRAGMA_UNROLL
        for (int iter_row = 0; iter_row < _::n_iters_row; ++iter_row) {
            const uint32_t row = row_offset + iter_row * _::block_height;
            
            if (half_mode) {
                // if (blockIdx.x == 0 && threadIdx.x == 0) printf("wi: %d, store acts from %d acc to [%d, %d]\n", warp_idx, iter_row, row, weights_col);

                // If T = T_ACC = __half then copy directly into activations shared memory
                wmma::store_matrix_sync((T_ACC*)act_shmem + row * _::shmem_stride + weights_col, result_frag[iter_row], _::shmem_stride, wmma::mem_row_major);
            } else {
                // Else we need to copy to auxillary shared memory and downcast it later
                wmma::store_matrix_sync(aux_shmem + row * _::acc_shmem_stride + weights_col, result_frag[iter_row], _::acc_shmem_stride, wmma::mem_row_major);
            }
        }

    }

#if false
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("OUTPUT SHMEM:\n");
        for (int i = 0; i < min(_::chunk_height, 16); ++i) {
            printf("%4d | ", i);
            for (int j = 0; j < _::chunk_width; ++j) {
                float x = act_shmem[i * _::shmem_stride + j];
                printf("%8.3f ", x);
            }
            printf("\n");
        }
        printf("\n");
    }
	__syncthreads();
#endif

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     for (int i = 0; i < CHUNK_HEIGHT; ++i) {
    //         printf("%4d | ", i);
    //         for (int j = 0; j < WIDTH; ++j) {
    //             float x = act_shmem[i * AUX_SHM_STRIDE + j];
    //             // x = max(x, 0);
    //             // x = (int)round((float)x / 256) - 128;
    //             // x = min(x, 127);
    //             printf("%8.3f ", x);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    if (!std::is_same<T, T_ACC>::value) {
        __syncthreads();

        // Convert int accumulator to T
        NumericConverter<T, CalculationPolicy>::convert(act_shmem, aux_shmem);
    }

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

        threadblock_store_output_static<CalculationPolicy>(out_intermediate_threadblock_this_layer, act_shmem);
	}
}


template <typename T, int WIDTH>
class FullyFusedQATMLP : public Network<T> {
public:
	FullyFusedQATMLP(uint32_t input_width, uint32_t output_width, uint32_t n_hidden_layers, Activation activation, Activation output_activation);

	void inference_mixed_precision_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override;

	std::unique_ptr<Context> forward_impl(cudaStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override;

	void backward_impl(
		cudaStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		EGradientMode param_gradients_mode = EGradientMode::Overwrite
	) override;

	void set_params_impl(T* params, T* inference_params, T* gradients) override;
	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override;

	GPUMatrix<T, RM>& input_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.front();
	}

	GPUMatrix<T, RM>& weight_matrix_at(bool inference, uint32_t idx) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_weight_matrix(bool inference) {
		auto& weight_matrices = inference ? m_weight_matrices_inference : m_weight_matrices;
		return weight_matrices.back();
	}

	GPUMatrix<T, RM>& input_gradient_matrix() {
		return m_gradient_matrices.front();
	}

	GPUMatrix<T, RM>& gradient_matrix_at(uint32_t idx) {
		return m_gradient_matrices.at(1 + idx);
	}

	GPUMatrix<T, RM>& output_gradient_matrix() {
		return m_gradient_matrices.back();
	}

	size_t n_params() const override {
		return m_total_n_params;
	}

	uint32_t input_width() const override {
		return m_input_width;
	}

	uint32_t padded_output_width() const override {
		return m_padded_output_width;
	}

	uint32_t output_width() const override {
		return m_output_width;
	}

	static uint32_t REQUIRED_ALIGNMENT() {
		return 16; // Uses 16x16x16 tensor ops
	}

	uint32_t required_input_alignment() const override {
		return REQUIRED_ALIGNMENT();
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		std::vector<std::pair<uint32_t, uint32_t>> result;
		for (auto& matrix : m_weight_matrices) {
			result.emplace_back(matrix.m(), matrix.n());
		}
		return result;
	}

	uint32_t width(uint32_t layer) const override {
		return WIDTH;
	}

	uint32_t num_forward_activations() const override {
		return m_n_hidden_layers;
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return {forward.hidden.at(layer).data(), CM};
	}

	json hyperparams() const override {
		return {
			{"otype", "FullyFusedQATMLP"},
			{"activation", to_string(m_activation)},
			{"output_activation", to_string(m_output_activation)},
			{"n_neurons", m_network_width},
			{"n_hidden_layers", m_n_hidden_layers},
		};
	}

private:
	struct ForwardContext : public Context {
		std::vector<GPUMatrix<T>> hidden;
		GPUMemoryArena::Allocation alloc;
	};

	std::unique_ptr<ForwardContext> allocate_forward_buffers(cudaStream_t stream, uint32_t batch_size);

	uint32_t m_n_hidden_layers;
	uint32_t m_n_hidden_matmuls;
	uint32_t m_input_width;
	uint32_t m_network_width;
	uint32_t m_output_width;
	uint32_t m_padded_output_width;

	Activation m_activation;
	Activation m_output_activation;

	// Storage of params
	std::vector<GPUMatrix<T, RM>> m_weight_matrices;
	std::vector<GPUMatrix<int8_t, RM>> m_weight_matrices_inference;
	size_t m_total_n_params;

	std::vector<GPUMatrix<T, RM>> m_gradient_matrices;
};

TCNN_NAMESPACE_END
