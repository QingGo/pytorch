#pragma once
#include <torch/csrc/inductor/cpp_wrapper/common.h>
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#include <torch/headeronly/util/Exception.h>

template <typename T>
[[maybe_unused]] static inline std::string getCudaDriverErrorString(T code) {
#if defined(USE_ROCM)
  return std::string("CUDA driver error code: ") +
      std::to_string(static_cast<int>(code));
#else
  const char* msg = nullptr;
  CUresult code_get_error = cuGetErrorString(static_cast<CUresult>(code), &msg);
  if (code_get_error != CUDA_SUCCESS) {
    return "CUDA driver error: invalid error code!";
  }
  return std::string("CUDA driver error: ") + std::string(msg);
#endif
}

#if defined(USE_ROCM)
#define CUDA_DRIVER_SUCCESS hipSuccess
#else
#define CUDA_DRIVER_SUCCESS CUDA_SUCCESS
#endif

#define CUDA_DRIVER_CHECK(EXPR)                                       \
  do {                                                                \
    auto code = EXPR;                                                 \
    STD_TORCH_CHECK(                                                  \
        code == CUDA_DRIVER_SUCCESS, getCudaDriverErrorString(code)); \
  } while (0)

[[maybe_unused]] static inline CUfunction loadKernel(
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    const std::optional<std::string>& cubinDir = std::nullopt) {
  if (cubinDir) {
    std::filesystem::path p1{*cubinDir};
    std::filesystem::path p2{filePath};
    filePath = (p1 / p2.filename()).string();
  }

  CUmodule mod = nullptr;
  CUfunction func = nullptr;
#if defined(USE_ROCM)
  CUDA_DRIVER_CHECK(hipModuleLoad(&mod, filePath.c_str()));
  CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
  if (sharedMemBytes > 0) {
    CUDA_DRIVER_CHECK(hipFuncSetAttribute(
        func, hipFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes));
  }
#else
  CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
  CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
  if (sharedMemBytes > 0) {
    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
  }
#endif
  return func;
}

[[maybe_unused]] static inline CUfunction loadKernel(
    const void* start,
    const std::string& funcName,
    uint32_t sharedMemBytes) {
#if defined(USE_ROCM)
  STD_TORCH_CHECK(
      false,
      "Embedded kernel binary load is not supported for ROCm cpp_wrapper");
#else
  CUmodule mod = nullptr;
  CUfunction func = nullptr;
  CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
  CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
  if (sharedMemBytes > 0) {
    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemBytes));
  }
  return func;
#endif
}

[[maybe_unused]] static inline void launchKernel(
    CUfunction func,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    void* args[],
    cudaStream_t stream) {
#if defined(USE_ROCM)
  int device = 0;
  CUDA_DRIVER_CHECK(hipGetDevice(&device));
  int warp_size = 0;
  CUDA_DRIVER_CHECK(
      hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device));

  CUDA_DRIVER_CHECK(hipModuleLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      warp_size * numWarps,
      1,
      1,
      sharedMemBytes,
      stream,
      args,
      nullptr));
#else
  CUDA_DRIVER_CHECK(cuLaunchKernel(
      func,
      gridX,
      gridY,
      gridZ,
      32 * numWarps,
      1,
      1,
      sharedMemBytes,
      stream,
      args,
      nullptr));
#endif
}

#include <torch/csrc/inductor/cpp_wrapper/lazy_triton_compile.h>

#ifdef TORCH_INDUCTOR_PRECOMPILE_HEADERS
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
