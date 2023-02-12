add_executable(test_qat_gemm test_qat_gemm.cu)
target_link_libraries(test_qat_gemm PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn cublas)