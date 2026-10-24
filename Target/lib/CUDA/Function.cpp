//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <utility>

using M::Target::CUDA::cudaErrorStatus;

static std::pair<llvm::SmallVector<CUjit_option, 2>,
                 llvm::SmallVector<uintptr_t, 2>>
setupCompilationOptions(const char *debugLevel, int32_t optimizationLevel) {
  static constexpr int kDefaultOptimizationLevel = 4;

  llvm::SmallVector<CUjit_option, 2> optionKind;
  llvm::SmallVector<uintptr_t, 2> optionValues;

  if (debugLevel != nullptr && std::strcmp(debugLevel, "full") == 0) {
    // The PTX compiler complains if debugLevel=full and optimization level is
    // something other than 0 or 4 (the default).
    assert(optimizationLevel < 0 || optimizationLevel == 0 ||
           optimizationLevel == kDefaultOptimizationLevel);

    optionKind.push_back(CU_JIT_GENERATE_DEBUG_INFO);
    optionValues.push_back(1);
  } else if (debugLevel != nullptr &&
             std::strcmp(debugLevel, "line-tables") == 0) {
    optionKind.push_back(CU_JIT_GENERATE_LINE_INFO);
    optionValues.push_back(1);
  }

  // The default optimization level for NVIDIA GPUs is 4.
  if (optimizationLevel < 0)
    optimizationLevel = kDefaultOptimizationLevel;

  // The Mojo compiler uses optimization level 3 as the highest optimization
  // level, but 4 is the highest for CUDA.
  if (optimizationLevel == 3)
    optimizationLevel = kDefaultOptimizationLevel;

  if (optimizationLevel != kDefaultOptimizationLevel) {
    optionKind.push_back(CU_JIT_OPTIMIZATION_LEVEL);
    optionValues.push_back(static_cast<uintptr_t>(optimizationLevel));
  }

  return {std::move(optionKind), std::move(optionValues)};
}

DriverResult<M_driver_runtime_bundle_t>
M_driver_runtime_bundle::load(M_driver_context &ctx,
                              M_driver_static_bundle_t bundle,
                              M_driver_bundle_compilation_options_t opts) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  auto [optionKind, optionValues] =
      setupCompilationOptions(opts->debug_level, opts->optimization_level);

  CUmodule module;
  if (CUresult r = cuda.cuModuleLoadDataEx(
          &module, bundle->mapped_data.data, optionKind.size(),
          optionKind.data(), reinterpret_cast<void **>(optionValues.data()));
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&ctx.device.driver, r, "cuModuleLoadDataEx"));

  auto runtimeBundle = new M_driver_runtime_bundle{.module = module};

  return runtimeBundle;
}

M_driver_status_t
M_driver_runtime_bundle::unload(M_driver_context &ctx,
                                M_driver_runtime_bundle_t bundle) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  if (CUresult r = cuda.cuModuleUnload(bundle->module); r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx.device.driver, r, "cuModuleUnload");

  delete bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_function_t>
M_driver_function::load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
                        const char *function_name, uint64_t function_name_len) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  CUfunction func;
  if (CUresult r =
          cuda.cuModuleGetFunction(&func, bundle->module, function_name);
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&ctx.device.driver, r,
                        "cuModuleGetFunction " + llvm::Twine(function_name)));

  if (CUresult r = cuda.cuFuncSetAttribute(
          func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
      r != CUDA_SUCCESS)
    return tl::unexpected(cudaErrorStatus(
        &ctx.device.driver, r,
        "cuFuncSetAttribute NON_PORTABLE_CLUSTER_SIZE_ALLOWED"));

  auto ptr = std::make_unique<M_driver_function>(
      M_driver_function{.function = func, .module = bundle->module});
  return ptr.release();
}

M_driver_status_t M_driver_function::unload(M_driver_context &ctx,
                                            M_driver_function_t func) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  delete func;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_function::getNativeHandle(M_driver_native_handle_t *native_handle) {
  if (native_handle == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *native_handle = reinterpret_cast<M_driver_native_handle_t>(module);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function::setAttribute(M_driver_context &ctx,
                                                  int32_t attribute,
                                                  int32_t value) {
  M::Target::CUDA::CUDALoader &cuda = ctx.device.driver.rawCuda;
  if (CUresult r = cuda.cuFuncSetAttribute(
          function, static_cast<CUfunction_attribute>(attribute), value);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx.device.driver, r, "cuFuncSetAttribute");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function::occupancyMaxActiveBlocks(
    M_driver_context &ctx, int32_t block_size,
    uint64_t dynamic_shared_memory_bytes, int32_t *num_blocks) {
  if (num_blocks == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M::Target::CUDA::CUDALoader &cuda = ctx.device.driver.rawCuda;
  int blocks = 0;
  if (CUresult r = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks, function, block_size,
          static_cast<size_t>(dynamic_shared_memory_bytes));
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx.device.driver, r,
                           "cuOccupancyMaxActiveBlocksPerMultiprocessor");
  *num_blocks = blocks;
  return M_DRIVER_STATUS_SUCCESS;
}
