#pragma once

#include <exception>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_include/kernel_compile_result.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/cpp_wrapper/common.h>
#if defined(USE_XPU)
#include <torch/csrc/inductor/cpp_wrapper/device_internal/xpu.h>
#else
#include <torch/csrc/inductor/cpp_wrapper/device_internal/cuda.h>
#endif

struct LazyTritonKernelSpec {
  const char* kernel_name;
  const char* kernel_source_key;
};

struct LazyTritonModuleState {
  PyObject* pending_kernels = nullptr;
};

struct LazyTritonKernelState {
  CUfunction function = nullptr;
  LazyKernelCompileResult compile_result;
  bool ready = false;
};

struct LazyTritonScratchBuffers {
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  RAIIAtenTensorHandle global_scratch_tensor;
  RAIIAtenTensorHandle profile_scratch_tensor;
};

// Forward declarations of the shared CUDA driver helpers defined in cuda.h.
// cuda.h includes this header after defining them so callers from the CUDA
// path resolve through inline definitions. The declarations exist so the
// templates below compile when this header is pulled in through XPU's wrapper
// header chain; XPU codegen does not instantiate launchLazyTritonKernel, so
// the unresolved names never reach the linker.
[[maybe_unused]] static inline CUfunction loadKernel(
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    const std::optional<std::string>& cubinDir);

[[maybe_unused]] static inline CUfunction loadKernel(
    const void* start,
    const std::string& funcName,
    uint32_t sharedMemBytes);

[[maybe_unused]] static inline void launchKernel(
    CUfunction func,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemBytes,
    void* args[],
    cudaStream_t stream);

// Python bridge symbols resolved on first use via loadLazyCompileFuncs().
// These are one-time-populated pointers into the Triton lazy-compile runtime;
// every caller already holds the GIL through py::gil_scoped_acquire_simple.
static PyObject* (*_THPVariable_Wrap)(const at::TensorBase&) = nullptr;
static int32_t (*_THPUtils_unpackInt)(PyObject*) = nullptr;
static PyObject* triton_lazy_compile_module = nullptr;
static PyObject* start_kernel_compile = nullptr;
static PyObject* run_triton_kernel_with_autotune = nullptr;

static inline void loadLazyCompileFuncs() {
  if (triton_lazy_compile_module == nullptr) {
    triton_lazy_compile_module =
        PyImport_ImportModule("torch._inductor.runtime.triton_lazy_compile");
    AOTI_TORCH_CHECK(
        triton_lazy_compile_module, "Failed to import triton_lazy_compile");

    start_kernel_compile = PyObject_GetAttrString(
        triton_lazy_compile_module, "start_kernel_compile");
    AOTI_TORCH_CHECK(
        start_kernel_compile, "Failed to get start_kernel_compile");

    run_triton_kernel_with_autotune = PyObject_GetAttrString(
        triton_lazy_compile_module, "run_triton_kernel_with_autotune");
    AOTI_TORCH_CHECK(
        run_triton_kernel_with_autotune,
        "Failed to get run_triton_kernel_with_autotune");

    RAIIPyObject guards_mod = PyImport_ImportModule("torch._C._dynamo.guards");
    AOTI_TORCH_CHECK(guards_mod, "Failed to import torch._C._dynamo.guards");

    RAIIPyObject wrap_addr =
        PyObject_GetAttrString(guards_mod, "_torchinductor_thp_variable_wrap");
    AOTI_TORCH_CHECK(
        wrap_addr, "Failed to get _torchinductor_thp_variable_wrap");
    _THPVariable_Wrap = reinterpret_cast<decltype(_THPVariable_Wrap)>(
        PyLong_AsVoidPtr(wrap_addr));
    AOTI_TORCH_CHECK(_THPVariable_Wrap, "THPVariable_Wrap not resolved");

    RAIIPyObject unpack_addr = PyObject_GetAttrString(
        guards_mod, "_torchinductor_thputils_unpack_int");
    AOTI_TORCH_CHECK(
        unpack_addr, "Failed to get _torchinductor_thputils_unpack_int");
    _THPUtils_unpackInt = reinterpret_cast<decltype(_THPUtils_unpackInt)>(
        PyLong_AsVoidPtr(unpack_addr));
    AOTI_TORCH_CHECK(_THPUtils_unpackInt, "THPUtils_unpackInt not resolved");
  }
}

static inline PyObject* getPendingKernelsForModule(
    LazyTritonModuleState* module_state) {
  AOTI_TORCH_CHECK(module_state, "Invalid lazy Triton module state");
  if (module_state->pending_kernels == nullptr) {
    module_state->pending_kernels = PyDict_New();
    AOTI_TORCH_CHECK(
        module_state->pending_kernels, "Failed to create pending kernels dict");
  }
  return module_state->pending_kernels;
}

[[noreturn]] static inline void throwPythonError(const char* context) {
  std::string msg(context);
  if (PyErr_Occurred()) {
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    RAIIPyObject type_obj(type);
    RAIIPyObject value_obj(value);
    RAIIPyObject traceback_obj(traceback);
    if (value_obj) {
      RAIIPyObject value_str = PyObject_Str(value_obj.get());
      if (value_str) {
        const char* str = PyUnicode_AsUTF8(value_str.get());
        if (str) {
          msg += ": ";
          msg += str;
        }
      }
    }
    PyErr_Clear();
  }
  TORCH_CHECK(false, msg);
}

// Helpers for reading attributes off the compile-result Python object.
// Every call is GIL-held (the caller acquired it before invoking these).
// Note: AOTI_TORCH_CHECK only accepts a single C-string message, so we can't
// include the attr name in the error; callers keep names localized instead.
static inline RAIIPyObject getRequiredAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = PyObject_GetAttrString(obj, attr);
  AOTI_TORCH_CHECK(val, "Failed to get attribute");
  return val;
}

static inline std::string getStringAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = getRequiredAttr(obj, attr);
  const char* str = PyUnicode_AsUTF8(val);
  AOTI_TORCH_CHECK(str, "Failed to decode string attribute");
  return str;
}

static inline int getIntAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = getRequiredAttr(obj, attr);
  return _THPUtils_unpackInt(val);
}

static inline int getOptionalIntAttr(
    PyObject* obj,
    const char* attr,
    int sentinel = -1) {
  RAIIPyObject val = getRequiredAttr(obj, attr);
  return (val.get() != Py_None) ? _THPUtils_unpackInt(val) : sentinel;
}

static inline std::vector<int> getIntListAttr(PyObject* obj, const char* attr) {
  RAIIPyObject val = getRequiredAttr(obj, attr);
  AOTI_TORCH_CHECK(val && PyList_Check(val.get()), "Expected list attribute");
  Py_ssize_t size = PyList_Size(val);
  std::vector<int> result;
  result.reserve(size);
  for (Py_ssize_t i = 0; i < size; i++) {
    result.push_back(_THPUtils_unpackInt(PyList_GetItem(val, i)));
  }
  return result;
}

static inline LazyKernelCompileResult extractCompileResult(PyObject* result) {
  LazyKernelCompileResult compile_result;
  compile_result.cubin_path = getStringAttr(result, "cubin_path");
  compile_result.mangled_name = getStringAttr(result, "mangled_name");
  compile_result.num_warps = getIntAttr(result, "num_warps");
  compile_result.shared_mem = getIntAttr(result, "shared_mem");
  compile_result.xblocks = getIntListAttr(result, "xblocks");
  compile_result.yblocks = getIntListAttr(result, "yblocks");
  compile_result.zblocks = getIntListAttr(result, "zblocks");
  compile_result.r0blocks = getIntListAttr(result, "r0blocks");
  compile_result.rsplit = getIntAttr(result, "rsplit");
  compile_result.rsplit_size = getIntAttr(result, "rsplit_size");
  compile_result.config_index = getOptionalIntAttr(result, "config_index");
  compile_result.global_scratch = getOptionalIntAttr(result, "global_scratch");
  compile_result.profile_scratch =
      getOptionalIntAttr(result, "profile_scratch");
  return compile_result;
}

template <typename T>
static inline PyObject* convertArgToPython(const T& arg) {
  using DecayedT = std::decay_t<T>;
  if constexpr (std::is_same_v<DecayedT, AtenTensorHandle>) {
    at::Tensor* tensor_ptr =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(arg);
    return _THPVariable_Wrap(*tensor_ptr);
  } else if constexpr (std::is_same_v<
                           DecayedT,
                           torch::aot_inductor::RAIIAtenTensorHandle>) {
    at::Tensor* tensor_ptr =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(arg.get());
    return _THPVariable_Wrap(*tensor_ptr);
  } else if constexpr (std::is_same_v<DecayedT, bool>) {
    PyObject* py_arg = arg ? Py_True : Py_False;
    Py_INCREF(py_arg);
    return py_arg;
  } else if constexpr (std::is_integral_v<DecayedT>) {
    return PyLong_FromLongLong(static_cast<long long>(arg));
  } else if constexpr (std::is_floating_point_v<DecayedT>) {
    return PyFloat_FromDouble(static_cast<double>(arg));
  } else {
    AOTI_TORCH_CHECK(false, "Invalid input type to convertArgToPython");
  }
}

template <typename... Args>
static inline LazyKernelCompileResult runTritonKernelWithAutotune(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    const std::string& kernel_source_key,
    void* stream,
    const Args&... kernel_args) {
  py::gil_scoped_acquire_simple acquire;
  loadLazyCompileFuncs();

  constexpr size_t num_args = sizeof...(Args);
  RAIIPyObject py_args_list = PyList_New(num_args);
  AOTI_TORCH_CHECK(py_args_list, "Failed to create args list");

  size_t idx = 0;
  auto add_arg = [&py_args_list, &idx](PyObject* py_arg) {
    AOTI_TORCH_CHECK(py_arg, "Failed to convert argument");
    if (PyList_SetItem(py_args_list, idx++, py_arg) != 0) {
      throwPythonError("Failed to set kernel argument");
    }
  };
  // Use array pack-expansion instead of a fold expression to avoid
  // hitting the compiler's expression-nesting limit when there are
  // hundreds of kernel arguments (e.g. combo kernels).
  int dummy[] = {0, (add_arg(convertArgToPython(kernel_args)), 0)...};
  (void)dummy;

  RAIIPyObject py_kernel_name = PyUnicode_FromString(kernel_name.c_str());
  RAIIPyObject py_source_key = PyUnicode_FromString(kernel_source_key.c_str());
  RAIIPyObject py_stream = PyLong_FromVoidPtr(stream);
  AOTI_TORCH_CHECK(
      py_kernel_name && py_source_key && py_stream,
      "Failed to create kernel call args");

  RAIIPyObject call_args = PyTuple_Pack(
      5,
      pending_kernels,
      py_kernel_name.get(),
      py_source_key.get(),
      py_stream.get(),
      py_args_list.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result =
      PyObject_CallObject(run_triton_kernel_with_autotune, call_args);
  if (!result) {
    throwPythonError("Failed to run kernel with autotuning");
  }

  return extractCompileResult(result);
}

static inline void startKernelCompile(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    const std::string& kernel_source_key) {
  py::gil_scoped_acquire_simple acquire;
  loadLazyCompileFuncs();

  RAIIPyObject py_name = PyUnicode_FromString(kernel_name.c_str());
  RAIIPyObject py_source_key = PyUnicode_FromString(kernel_source_key.c_str());
  AOTI_TORCH_CHECK(py_name && py_source_key, "Failed to create Python args");

  RAIIPyObject call_args =
      PyTuple_Pack(3, pending_kernels, py_name.get(), py_source_key.get());
  AOTI_TORCH_CHECK(call_args, "Failed to create call args");

  RAIIPyObject result = PyObject_CallObject(start_kernel_compile, call_args);
  if (!result) {
    throwPythonError("Failed to start kernel compilation");
  }
}

static inline void startKernelCompileBestEffort(
    PyObject* pending_kernels,
    const std::string& kernel_name,
    const std::string& kernel_source_key) {
  try {
    startKernelCompile(pending_kernels, kernel_name, kernel_source_key);
  } catch (const std::exception&) {
    PyErr_Clear();
  }
}

static inline void startKernelCompilesForModule(
    LazyTritonModuleState* module_state,
    const LazyTritonKernelSpec* const* kernel_specs,
    size_t num_kernel_specs) {
  py::gil_scoped_acquire_simple acquire;
  loadLazyCompileFuncs();
  PyObject* pending_kernels = getPendingKernelsForModule(module_state);
  for (size_t i = 0; i < num_kernel_specs; ++i) {
    const LazyTritonKernelSpec* kernel_spec = kernel_specs[i];
    AOTI_TORCH_CHECK(kernel_spec, "Invalid lazy Triton kernel spec");
    startKernelCompileBestEffort(
        pending_kernels,
        kernel_spec->kernel_name,
        kernel_spec->kernel_source_key);
  }
}

static inline void allocateLazyTritonScratchBuffer(
    int64_t scratch_size,
    int32_t device_idx,
    CUdeviceptr* scratch_ptr,
    RAIIAtenTensorHandle* scratch_tensor) {
  AOTI_TORCH_CHECK(scratch_ptr, "Invalid lazy Triton scratch pointer");
  AOTI_TORCH_CHECK(scratch_tensor, "Invalid lazy Triton scratch tensor");
  if (scratch_size <= 0) {
    return;
  }

  int64_t scratch_sizes[] = {scratch_size};
  int64_t scratch_strides[] = {1};
  AtenTensorHandle scratch_handle;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      1,
      scratch_sizes,
      scratch_strides,
      aoti_torch_dtype_uint8(),
      aoti_torch_device_type_cuda(),
      device_idx,
      &scratch_handle));
  *scratch_tensor = RAIIAtenTensorHandle(scratch_handle);
  *scratch_ptr = reinterpret_cast<CUdeviceptr>(scratch_tensor->data_ptr());
}

template <typename... Args>
static inline bool ensureLazyTritonKernelReady(
    LazyTritonModuleState* module_state,
    const LazyTritonKernelSpec* kernel_spec,
    LazyTritonKernelState* kernel_state,
    cudaStream_t stream,
    const Args&... kernel_args) {
  AOTI_TORCH_CHECK(kernel_spec, "Invalid lazy Triton kernel spec");
  AOTI_TORCH_CHECK(kernel_state, "Invalid lazy Triton kernel state");

  if (kernel_state->ready) {
    return false;
  }

  PyObject* pending_kernels = getPendingKernelsForModule(module_state);
  startKernelCompile(
      pending_kernels,
      kernel_spec->kernel_name,
      kernel_spec->kernel_source_key);
  kernel_state->compile_result = runTritonKernelWithAutotune(
      pending_kernels,
      kernel_spec->kernel_name,
      kernel_spec->kernel_source_key,
      stream,
      kernel_args...);
  kernel_state->function = loadKernel(
      kernel_state->compile_result.cubin_path,
      kernel_state->compile_result.mangled_name,
      kernel_state->compile_result.shared_mem,
      std::nullopt);
  kernel_state->ready = true;
  return true;
}

template <typename... Args>
static inline void launchLazyTritonKernel(
    const LazyTritonKernelState* kernel_state,
    int32_t device_idx,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    cudaStream_t stream,
    Args... kernel_args) {
  AOTI_TORCH_CHECK(kernel_state, "Invalid lazy Triton kernel state");
  AOTI_TORCH_CHECK(kernel_state->function, "Lazy Triton kernel is not loaded");

  LazyTritonScratchBuffers scratch_buffers;
  allocateLazyTritonScratchBuffer(
      kernel_state->compile_result.global_scratch,
      device_idx,
      &scratch_buffers.global_scratch,
      &scratch_buffers.global_scratch_tensor);
  allocateLazyTritonScratchBuffer(
      kernel_state->compile_result.profile_scratch,
      device_idx,
      &scratch_buffers.profile_scratch,
      &scratch_buffers.profile_scratch_tensor);

  void* launch_args[] = {
      kernel_args...,
      &scratch_buffers.global_scratch,
      &scratch_buffers.profile_scratch};
  launchKernel(
      kernel_state->function,
      grid_x,
      grid_y,
      grid_z,
      kernel_state->compile_result.num_warps,
      kernel_state->compile_result.shared_mem,
      launch_args,
      stream);
}
