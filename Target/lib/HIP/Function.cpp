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
#include "Driver.h"
#include "Error.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_runtime_bundle_t>
M_driver_runtime_bundle::load(M_driver_context &ctx,
                              M_driver_static_bundle_t bundle,
                              M_driver_bundle_compilation_options_t opts) {
  if (bundle == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = ctx.getDevice().deviceMethodPreamble(); !result)
    return tl::unexpected(result.error());

  llvm::StringRef fileType(bundle->file_type, bundle->file_type_len);
  if (fileType != "hsaco")
    return tl::unexpected(M_DRIVER_STATUS_INCOMPATIBLE);

  hipModule_t module;
  if (hipError_t r = hip.hipModuleLoadData(&module, bundle->mapped_data.data);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&device.getDriver(), r, "hipModuleLoadData"));

  return new M_driver_runtime_bundle{.module = std::move(module)};
}

M_driver_status_t
M_driver_runtime_bundle::unload(M_driver_context &ctx,
                                M_driver_runtime_bundle_t bundle) {
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = device.deviceMethodPreamble(); !result)
    return result.error();

  if (hipError_t r = hip.hipModuleUnload(bundle->module); r != hipSuccess)
    return hipErrorStatus(&device.getDriver(), r, "hipModuleUnload");

  delete bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_function_t>
M_driver_function::load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
                        const char *function_name, uint64_t function_name_len) {
  if (bundle == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);

  M_driver_device &device = ctx.getDevice();
  HIPApi &hip = device.getDriver().hip();

  if (auto result = device.deviceMethodPreamble(); !result)
    return tl::unexpected(result.error());

  hipFunction_t func;
  if (hipError_t r =
          hip.hipModuleGetFunction(&func, bundle->module, function_name);
      r != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&device.getDriver(), r,
                       "hipModuleGetFunction " + llvm::Twine(function_name)));

  return new M_driver_function{.function = std::move(func)};
}

M_driver_status_t M_driver_function::unload(M_driver_function_t func) {
  if (func == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;

  delete func;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function::occupancyMaxActiveBlocks(
    M_driver_context &ctx, int32_t block_size,
    uint64_t dynamic_shared_memory_bytes, int32_t *num_blocks) {
  if (num_blocks == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = ctx.getDevice().getDriver();
  int blocks = 0;
  if (hipError_t r =
          driver.hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks, function, block_size,
              static_cast<size_t>(dynamic_shared_memory_bytes));
      r != hipSuccess)
    return hipErrorStatus(&driver, r,
                          "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
  *num_blocks = blocks;
  return M_DRIVER_STATUS_SUCCESS;
}
