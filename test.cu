#include <mma.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdio.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/random.h>

using namespace nvcuda;

/*
 *
 * FUSED KERNEL LAUNCH:
 *     - num blocks            = batch_size / chunk_height
 *     - num threads per block = n_blocks warps (32 threads in each warp)
 *     - shared memory         = chunk_size * (sizeof(T) + sizeof(T_OUT)) (represents single chunk)
 *
 * INPUT / OUTPUT MATRIX:
 *                                width (= 16 * n_blocks)
 *                          ◄─────────────────────────────────►
 *                                                    
 *                             16                     ┌────────┐ ◄───── block (16x16 matrix)
 *                          ◄──────►
 *                         ┌────────┬────────┬────────┬────────┐ ─┐                     ─┐
 *                  ▲    ▲ │        │        │        │        │  │                      │
 *                  │ 16 │ │        │        │        │        │  │                      │
 *                  │    │ │        │        │        │        │  │◄─── stripe           │
 *                  │    ▼ │        │        │        │        │  │ (16 x width matrix)  │
 *                  │      ├────────┼────────┼────────┼────────┤ ─┘                      │
 *                  │      │        │        │        │        │                         │
 *                  │      │        │        │        │        │                         │
 *                  │      │        │        │        │        │                         │
 *                  │      │        │        │        │        │                         │
 *     chunk_height │      ├────────┼────────┼────────┼────────┤                         │◄── chunk
 *                  │      │        │        │        │        │                         │
 * (= 16 * n_iters) │      │        │        │        │        │                         │ (chunk_height x width matrix,
 *                  │      │        │        │        │        │                         │  processed by signle threadblock)
 *                  │      │        │        │        │        │                         │
 *                  │      ├────────┼────────┼────────┼────────┤                         │
 *                  │      │        │        │        │        │                         │
 *                  │      │        │        │        │        │                         │
 *                  │      │        │        │        │        │                         │
 *                  ▼      │        │        │        │        │                         │
 *                         ├────────┼────────┼────────┼────────┤                        ─┘
 *                         │                ...                │
 *                         │                                   │
 *                         │                                   │
 *                         ├────────┬────────┬────────┬────────┤
 *                         │        │        │        │        │
 *                         │        │        │        │        │
 *                         │        │        │        │        │
 *                         │        │        │        │        │
 *                         └────────┴────────┴────────┴────────┘
 */

template<typename T>
void head(const std::vector<T>& data, int width, int height = 16) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << std::setw(5) << (float)data[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

template<typename T>
__global__
void fill_rand(uint32_t size, T* out, tcnn::default_rng_t rng) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i >= size) {
        return;
    }

    rng.advance(i);

    if (std::is_same<T, int8_t>::value) {
        out[i] = (uint8_t)(rng.next_uint() & 0xFF);
    } else {
        out[i] = (T)((rng.next_float() - 0.5) * 2);
    }
}

template<typename T>
__global__
void fill_val(uint32_t size, T* out, T val) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    out[i] = val;
}

#define DEBUG_PRINT 0

// template <typename T, typename T_OUT, uint32_t width, uint32_t n_iters>
// __device__ inline
// void copy(T* __restrict__ dst, const T_OUT* __restrict__ src);

// template <uint32_t width, uint32_t n_iters>
// __device__ inline
// void copy<__half, __half, width, n_iters>(__half* __restrict__ dst, const __half* __restrict__ src) {
//     throw std::runtime_error("Not implemented for");
// }

// template <uint32_t width, uint32_t n_iters>
// __device__ inline
// void copy<__half, float, width, n_iters>(__half* __restrict__ dst, const float* __restrict__ src) {
//     constexpr uint32_t warp_size = 32;
//     constexpr uint32_t n_blocks = width / 16;
//     constexpr uint32_t chunk_height = n_iters * 16;
//     constexpr uint32_t chunk_size = chunk_height * width;

//     constexpr uint32_t dst_skew = 16 / sizeof(__half);
//     constexpr uint32_t src_skew = 16 / sizeof(float);

//     uint32_t li = threadIdx.x;
//     uint32_t wi = threadIdx.y;

//     #pragma unroll
//     for (int offset = 0; offset < chunk_size; offset += warp_size * n_blocks) {
//         uint32_t col = (offset + li) % width;
//         uint32_t row = (offset + wi * warp_size + li) / width;

//         dst[row * dst_skew + col] = (__half)src[row * src_skew + col];
//     }
// }

#define DEFINE_COMMON_CONSEXPR                                      \
    constexpr uint32_t SKEW = 16 / sizeof(T);                       \
    constexpr uint32_t BLOCK_SIZE = 16;                             \
    constexpr uint32_t N_BLOCKS = WIDTH / BLOCK_SIZE;               \
    constexpr uint32_t N_WARPS = N_BLOCKS;                          \
    constexpr uint32_t WARP_SIZE = 32;                              \
    constexpr uint32_t CHUNK_SIZE = (BLOCK_SIZE * N_ITERS) * WIDTH; \
    constexpr uint32_t STRIDE = WIDTH + SKEW;

/**
 * Copies chunk of size (16 * N_ITERS x WIDTH) from input_threadblock to act_shmem
 */ 
template <typename T, int WIDTH, int N_ITERS, typename T_COPY=int4>
__device__ void threadblock_load_input_static(T* __restrict__ act_shmem, const T* __restrict__ input_threadblock) {
    DEFINE_COMMON_CONSEXPR

    constexpr uint32_t COPY_STRIDE = sizeof(T_COPY) / sizeof(T);
    constexpr uint32_t STEP = N_WARPS * WARP_SIZE * COPY_STRIDE;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t col = (li * COPY_STRIDE) % WIDTH;
	const uint32_t row = (li + wi * WARP_SIZE) * STRIDE / WIDTH;


    // each lane (of WARP_SIZE lanes) copies COPY_STRIDE elements in single vector uint4 operation (or different T_COPY type, if specified)
    // each warp (of N_WARPS warps) copies WARP_SIZE * COPY_STRIDE elements
    // full threadblock copies then N_WARPS * WARP_SIZE * COPY_STRIDE elements at once, 
    // then repeats untill all of CHUNK_SIZE elements are copied.
	TCNN_PRAGMA_UNROLL
	for (int offset = 0; offset < CHUNK_SIZE; offset += STEP) {
		*(T_COPY*)&act_shmem[offset + row * STRIDE + col] = *(T_COPY*)&input_threadblock[offset + row * WIDTH + col];
	}
}

/**
 * Copies chunk of size (16 * N_ITERS x WIDTH) from act_shmem to output_threadblock
 */ 
template <typename T, int WIDTH, int N_ITERS, typename T_COPY=int4>
__device__ void threadblock_store_output_static(T* __restrict__ output_threadblock, const T* __restrict__ act_shmem) {
    DEFINE_COMMON_CONSEXPR

    constexpr uint32_t COPY_STRIDE = sizeof(T_COPY) / sizeof(T);
    constexpr uint32_t STEP = N_WARPS * WARP_SIZE * COPY_STRIDE;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t col = (li * COPY_STRIDE) % WIDTH;
	const uint32_t row = (li + wi * WARP_SIZE) * STRIDE / WIDTH;

	TCNN_PRAGMA_UNROLL
	for (int offset = 0; offset < CHUNK_SIZE; offset += STEP) {
		*(T_COPY*)&output_threadblock[offset + row * WIDTH + col] = *(T_COPY*)&act_shmem[offset + row * STRIDE + col];
	}
}


template<uint32_t WIDTH, uint32_t N_ITER>
__device__ void rowwise_sum(int32_t* __restrict__ act_shmem) {

}


template <typename T, int WIDTH, int N_ITERS, bool BACKWARD=false>
__device__ void qat_threadblock_layer(
    Activation activation,
    T* __restrict__ act_shmem,
    const T* __restrict__ weights_this_layer,
    T* __restrict__ out_intermediate_threadblock_this_layer,
    const T* __restrict__ activation_aux = nullptr
) {
    DEFINE_COMMON_CONSEXPR
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

	using namespace nvcuda;

	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, T, weights_layout_t> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, T> result_frag[N_ITERS];

	// Indices
	const uint32_t wi = threadIdx.y; // index in block ("warp index")
	const uint32_t weights_col = BLOCK_SIZE * wi;

	__syncthreads();

	// Load N_BLOCKS chunks of weights from global memory into registers.
	TCNN_PRAGMA_UNROLL
	for (uint32_t block = 0; block < N_BLOCKS; ++block) {
        const uint32_t row = block * BLOCK_SIZE;

		if (BACKWARD) {
			// If we're performing the backward pass, additional index swizzling is needed to
			// load the weights in transposed form.
			wmma::load_matrix_sync(weights_frag[block], weights_this_layer + row * WIDTH + weights_col, WIDTH);
		} else {
			wmma::load_matrix_sync(weights_frag[block], weights_this_layer + row + weights_col * WIDTH, WIDTH);
		}
	}

    // Multiply N_ITERS blocks of size (16 x WIDTH) by weights + apply activation
	TCNN_PRAGMA_UNROLL
	for (int iter = 0; iter < N_ITERS; ++iter) {
		const uint32_t row = iter * BLOCK_SIZE;
        
        wmma::fill_fragment(result_frag[iter], 0.0f);

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
            const uint32_t col = i * BLOCK_SIZE;

			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			wmma::load_matrix_sync(act_frag, act_shmem + row * STRIDE + col, STRIDE);
			wmma::mma_sync(result_frag[iter], act_frag, weights_frag[i], result_frag[iter]);
		}

		// Activation
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			wmma::load_matrix_sync(act_frag, activation_aux + row * WIDTH + weights_col, WIDTH);
			warp_activation_backward<T>(activation, result_frag[iter], act_frag, result_frag[iter]);
		} else {
			warp_activation<T>(activation, result_frag[iter], result_frag[iter]);
		}
	}

	__syncthreads();

    // Copy fragments back to shared memory
	TCNN_PRAGMA_UNROLL
	for (int iter = 0; iter < N_ITERS; ++iter) {
		const uint32_t row = iter * BLOCK_SIZE;

		wmma::store_matrix_sync(act_shmem + row * STRIDE + weights_col, result_frag[iter], DTRIDE, wmma::mem_row_major);
	}

    // Apply fake quantization
    if (!BACKWARD) {
        // TODO Fake quantization
    }

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

        threadblock_store_output_static(out_intermediate_threadblock_this_layer, act_shmem);
	}
}

template <int WIDTH, int N_ITERS, bool BACKWARD=false>
__device__ void qat_threadblock_inference_layer(
    Activation activation,
    uint8_t* __restrict__ act_shmem,
    const uint8_t* __restrict__ weights_this_layer,
    uint8_t* __restrict__ out_intermediate_threadblock_this_layer,
    const uint8_t* __restrict__ activation_aux = nullptr
) {
    DEFINE_COMMON_CONSEXPR

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, T, weights_layout_t> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, T> result_frag[N_ITERS];

	// Indices
	// const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	// const uint32_t lane_offset = (8 * li) % WIDTH;
	// const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	// const uint32_t weights_col = 16 * wi;

	__syncthreads();

	// Load N_BLOCKS chunks of weights from global memory into registers.
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_BLOCKS; ++i) {
        const uint32_t row = i * BLOCK_SIZE;

        // quantized threadblock layer runs only forward
		wmma::load_matrix_sync(weights_frag[i], weights_this_layer + row + weights_col * WIDTH, WIDTH);
	}

	TCNN_PRAGMA_UNROLL
	for (int iter = 0; iter < N_ITERS; ++iter) {
		const uint32_t row = iter * BLOCK_SIZE;
        
        wmma::fill_fragment(result_frag[iter], 0.0f);

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
            const uint32_t col = i * BLOCK_SIZE;

			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			wmma::load_matrix_sync(act_frag, act_shmem + row * STRIDE + col, STRIDE);
			wmma::mma_sync(result_frag[iter], act_frag, weights_frag[i], result_frag[iter]);
		}

		// Activation
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			wmma::load_matrix_sync(act_frag, activation_aux + row * WIDTH + weights_col, WIDTH);
			warp_activation_backward<T>(activation, result_frag[iter], act_frag, result_frag[iter]);
		} else {
			warp_activation<T>(activation, result_frag[iter], result_frag[iter]);
		}
	}

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (int iter = 0; iter < N_ITERS; ++iter) {
		const uint32_t row = iter * BLOCK_SIZE;

		wmma::store_matrix_sync(act_shmem + row * STRIDE + weights_col, result_frag[iter], DTRIDE, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

        threadblock_store_output_static(out_intermediate_threadblock_this_layer, act_shmem);
	}
}


template <typename T, typename T_OUT, uint32_t width, uint32_t n_iters, uint32_t n_layers>
__global__
void matmul(
    const T* __restrict__ in,
    const T* __restrict__ weights,
    T* __restrict__ out
) {
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t n_blocks = width / 16;
    constexpr uint32_t stripe_height = 16;
    constexpr uint32_t block_width = 16;
    constexpr uint32_t chunk_height = stripe_height * n_iters;
    constexpr uint32_t chunk_size = chunk_height * width;
    constexpr uint32_t weights_size = width * width;

    constexpr uint32_t copy_stride = sizeof(int4) / sizeof(T);
    // constexpr uint32_t copy_stride_out = sizeof(int4) / sizeof(T_OUT);

    constexpr uint32_t shmem_skew = 16 / sizeof(T);
    constexpr uint32_t shmem_acc_skew = 16 / sizeof(T_OUT);

    constexpr uint32_t shmem_stride = (width + shmem_skew);
    constexpr uint32_t shmem_acc_stride = (width + shmem_acc_skew);

    // static_assert(chunk_size % (32 * copy_stride) == 0);

    __shared__ T act_shmem[chunk_height * shmem_stride];
    __shared__ T_OUT shmem_acc[chunk_height * shmem_acc_stride];
    
    const uint32_t li = threadIdx.x;                      // lane index, 0..31
    const uint32_t wi = threadIdx.y;                      // warp index, 0..(n_blocks-1)
    const uint32_t chunk_row = blockIdx.x * chunk_height; // row in input matrix where the chunk starts

    threadblock_load_input_static(act_shmem, input + chunk_row * WIDTH);
    
    // #pragma unroll
    // for (int offset = 0; offset < chunk_size; offset += copy_stride * warp_size * n_blocks) {
    //     uint32_t col = (offset + li * copy_stride) % width;
    //     uint32_t row = (offset + (li + wi * warp_size) * copy_stride) / width;

    //     *(uint4*)&shmem[row * shmem_stride + col] = *(uint4*)&in[(chunk_row + row) * width + col];
    // }

#if DEBUG_PRINT
    constexpr uint32_t print_threadblock = 0;

    if (li == 0 && wi == 0 && blockIdx.x == print_threadblock) {
        printf("shmem on start:\n");
        for (int i = 0; i < chunk_height; ++i) {
            for (int j = 0; j < width; ++j) {
                printf("%4d ", shmem[i * width + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    __syncthreads();
#endif

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> mat_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> mat_b[n_blocks];
    wmma::fragment<wmma::accumulator, 16, 16, 16, T_OUT> acc[n_iters];

    uint32_t output_col = wi * block_width;

    #pragma unroll
    for (int layer = 0; layer < n_layers; ++layer) {
        const T* layer_weights = weights + layer * weights_size;

        #pragma unroll
        for (int i = 0; i < n_blocks; ++i) {
            uint32_t row = i * stripe_height;
            uint32_t col = wi * block_width;

            wmma::load_matrix_sync(mat_b[i], layer_weights + row + col * width, width);
        }

        #pragma unroll
        for (uint32_t iter = 0; iter < n_iters; ++iter) {
            uint32_t row = iter * stripe_height;

            wmma::fill_fragment(acc[iter], (T_OUT)0);

            for (int block = 0; block < n_blocks; ++block) {
                uint32_t col = block * block_width;

                wmma::load_matrix_sync(mat_a, &act_shmem[row * shmem_stride + col], shmem_stride);
                wmma::mma_sync(acc[iter], mat_a, mat_b[block], acc[iter]);
            }
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t iter = 0; iter < n_iters; ++iter) {
            uint32_t row = iter * stripe_height;

            if (std::is_same<T, T_OUT>::value) {
                wmma::store_matrix_sync(&act_shmem[row * shmem_stride + output_col], acc[iter], shmem_stride, wmma::mem_row_major);
            } else {
                wmma::store_matrix_sync(&shmem_acc[row * shmem_acc_stride + output_col], acc[iter], shmem_acc_stride, wmma::mem_row_major);
            }
        }
        
        if (!std::is_same<T, T_OUT>::value && layer != n_layers - 1) {
            __syncthreads();

            copy<T, T_OUT, width, n_iters>(act_shmem, shmem_acc);

        //     if (std::is_same<T, T_OUT>::value) {
        //         #pragma unroll
        //         for (int offset = 0; offset < chunk_size; offset += warp_size * n_blocks * copy_stride) {
        //             uint32_t col = (offset + li * copy_stride) % width;
        //             uint32_t row = (offset + (wi * warp_size + li) * copy_stride) / width;

        //             *(uint4*)&shmem[row * shmem_stride + col] = *(uint4*)&shmem_acc[row * shmem_acc_stride + col];
        //         }
        //     } else if (std::is_same<T, std::int8_t>::value) {
        //         #pragma unroll
        //         for (int offset = 0; offset < chunk_size; offset += warp_size * n_blocks * 4) {
        //             uint32_t col = (offset + li * 4) % width;
        //             uint32_t row = (offset + (wi * warp_size + li) * 4) / width;

        //             uint32_t word = shmem_acc[row * shmem_acc_stride + col] & 0xFF;
        //             word |= (shmem_acc[row * shmem_acc_stride + col + 1] & 0xFF) << 8;
        //             word |= (shmem_acc[row * shmem_acc_stride + col + 2] & 0xFF) << 16;
        //             word |= (shmem_acc[row * shmem_acc_stride + col + 3] & 0xFF) << 24;

        //             *(uint32_t*)&shmem[row * shmem_stride + col] = word;
        //         }
        //     } else {
        //         #pragma unroll
        //         for (int offset = 0; offset < chunk_size; offset += warp_size * n_blocks) {
        //             uint32_t col = (offset + li) % width;
        //             uint32_t row = (offset + wi * warp_size + li) / width;

        //             shmem[row * shmem_stride + col] = (T)shmem_acc[row * shmem_acc_stride + col];
        //         }
        //     }
        }

#if DEBUG_PRINT
        if (li == 0 && wi == 0 && blockIdx.x == print_threadblock) {
            printf("shmem after %d layer:\n", layer);
            for (int i = 0; i < chunk_height; ++i) {
                for (int j = 0; j < width; ++j) {
                    printf("%4d ", shmem_acc[i * shmem_acc_stride + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        __syncthreads();
#endif

    }

    __syncthreads();

    #pragma unroll
    for (int offset = 0; offset < chunk_size; offset += warp_size * n_blocks * copy_stride) {
        uint32_t col = (offset + li * copy_stride) % width;
        uint32_t row = (offset + (li + wi * warp_size) * copy_stride) / width;

        *(uint4*)&out[(chunk_row + row) * width + col] = *(uint4*)&act_shmem[row * shmem_stride + col];
    }
}

template <typename T, typename T_OUT, uint32_t width>
void test(uint64_t batch_size) {
    constexpr uint32_t n_iters = 4;
    constexpr uint32_t n_blocks = width / 16;
    constexpr uint32_t chunk_height = 16 * n_iters;
    // constexpr uint32_t chunk_size = chunk_height * width;
    constexpr uint32_t n_layers = 8;

    uint64_t weights_size = width * width * n_layers;
    uint64_t act_size = batch_size * width;

    uint64_t weights_size_bytes = sizeof(T) * weights_size;
    uint64_t in_size_bytes = sizeof(T) * act_size;
    uint64_t out_size_bytes = sizeof(T) * act_size;

    T* in = nullptr;
    T* weights = nullptr;
    T* out = nullptr;

    CUDA_CHECK_THROW(cudaMalloc((void**)&weights, weights_size_bytes));
    CUDA_CHECK_THROW(cudaMalloc((void**)&in, in_size_bytes));
    CUDA_CHECK_THROW(cudaMalloc((void**)&out, out_size_bytes));

    std::cout << width << " " << batch_size << std::endl;

    tcnn::default_rng_t rng;

    // std::vector<T> weights_cpu(weights_size);
    // std::vector<T> in_cpu(act_size);
    // CUDA_CHECK_THROW(cudaMemcpy(in_cpu.data(), in, in_cpu.size() * sizeof(T), cudaMemcpyDeviceToHost));
    // CUDA_CHECK_THROW(cudaMemcpy(weights_cpu.data(), weights, weights_cpu.size() * sizeof(T), cudaMemcpyDeviceToHost));
    // head(in_cpu, width, batch_size);
    // head(weights_cpu, width, width);

    dim3 blocks{(uint32_t)(batch_size / (uint64_t)chunk_height), 1, 1};
    dim3 threads{32, n_blocks, 1};

    for (int i = 0; i < 10; ++i) {
        rng.advance();
        fill_rand<T><<<act_size / 128, 128>>>(act_size, in, rng);

        rng.advance();
        fill_rand<T><<<weights_size / 128, 128>>>(weights_size, weights, rng);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        matmul<T, T_OUT, width, n_iters, n_layers><<<blocks, threads>>>(in, weights, out);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << std::setw(20) << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1'000'000 << " ms" << std::endl;
    }
    std::cout << std::endl;

    // std::vector<T_OUT> out_cpu(act_size);
    // CUDA_CHECK_THROW(cudaMemcpy(out_cpu.data(), out, out_cpu.size() * sizeof(T_OUT), cudaMemcpyDeviceToHost));
    // head(out_cpu, width, batch_size);

    CUDA_CHECK_THROW(cudaFree((void*)weights));
    CUDA_CHECK_THROW(cudaFree((void*)in));
    CUDA_CHECK_THROW(cudaFree((void*)out));
}

int main(int argc, char** argv) {
    uint64_t batch_size = 1 << 23;
    constexpr uint32_t width = 64;

    // uint32_t batch_size = std::atoi(argv[1]);
    // batch_size = (batch_size / 512) * 512;

    // try {
    //     std::cout << "int8 int32" << std::endl;
    //     test<int8_t, int32_t, width>(batch_size);
    // } catch (const std::runtime_error& exc) {
    //     std::cout << "failed " << exc.what() << std::endl;
    // }

    // try {
    //     std::cout << "half half" << std::endl;
    //     test<__half, __half, width>(batch_size);
    // } catch (const std::runtime_error& exc) {
    //     std::cout << "failed " << exc.what() << std::endl;
    // }

    // // try {
    //     std::cout << "half float" << std::endl;
    //     test<__half, float, width>(batch_size);
    // } catch (const std::runtime_error& exc) {
    //     std::cout << "failed " << exc.what() << std::endl;
    // }
    
}