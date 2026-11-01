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
#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"

#include <algorithm>

using namespace llvm;

[[nodiscard]] DriverResult<M_driver_runtime_bundle_t>
M_driver_runtime_bundle::load(M_driver_context_t context,
                              M_driver_static_bundle_t static_bundle,
                              M_driver_bundle_compilation_options_t opts) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mDevice = context->device->mDevice;
  if (mDevice == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);

  NS::SharedPtr<MTL::Library> library;

  // Detect if input is binary or MSL source code
  bool isBinary = false;
  if (static_bundle->mapped_data.size >= 4) {
    // Check if it's NOT MSL text (which would start with valid ASCII/UTF-8)
    // Look for non-printable characters in the first few bytes
    isBinary = false;
    for (size_t i = 0;
         i < std::min(size_t(8), size_t(static_bundle->mapped_data.size));
         ++i) {
      unsigned char c = static_bundle->mapped_data.data[i];
      if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
        isBinary = true;
        break;
      }
    }
  }

  if (!isBinary) {
    // Compile from MSL source code
    NS::String *sourceString = NS::String::string(
        reinterpret_cast<const char *>(static_bundle->mapped_data.data),
        NS::UTF8StringEncoding);
    NS::SharedPtr<MTL::CompileOptions> options =
        NS::TransferPtr(MTL::CompileOptions::alloc()->init());

    NS::Error *error = nullptr;
    library = NS::TransferPtr(
        mDevice->newLibrary(sourceString, options.get(), &error));
    if (error)
      return tl::unexpected(context->device->driver->status.create(
          Twine("Failed to create Metal library (compiling MSL to AIR): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding)));
  } else {
    // Load from binary metallib data
    dispatch_data_t binaryData = dispatch_data_create(
        static_bundle->mapped_data.data, static_bundle->mapped_data.size,
        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NS::Error *error = nullptr;
    library = NS::TransferPtr(mDevice->newLibrary(binaryData, &error));
    if (error)
      return tl::unexpected(context->device->driver->status.create(
          Twine("Failed to create Metal library (loading binary/metallib): ") +
          error->localizedDescription()->cString(NS::UTF8StringEncoding)));

    // Release the dispatch data
    dispatch_release(binaryData);
  }

  return new M_driver_runtime_bundle{.library = library};
}

[[nodiscard]] DriverResult<M_driver_function_t>
M_driver_function::create(M_driver_context_t context,
                          M_driver_runtime_bundle_t bundle,
                          llvm::StringRef name) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Get the Metal device
  MTL::Device *mDevice = context->device->mDevice;
  if (mDevice == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);

  // Get the function from the library
  NS::String *funcName =
      NS::String::string(name.str().c_str(), NS::UTF8StringEncoding);
  NS::SharedPtr<MTL::Function> mFunction =
      NS::TransferPtr(bundle->library->newFunction(funcName));
  if (!mFunction) {
    return tl::unexpected(context->device->driver->status.create(
        "Failed to create Metal function: " + Twine(name)));
  }

  // Create a compute pipeline state with reflection to get argument info
  NS::SharedPtr<MTL::ComputePipelineDescriptor> pipelineDesc =
      NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipelineDesc->setComputeFunction(mFunction.get());
  pipelineDesc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);

  // Create pipeline with reflection to get argument count
  MTL::ComputePipelineReflection *reflection = nullptr;
  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
      NS::TransferPtr(mDevice->newComputePipelineState(
          pipelineDesc.get(), MTL::PipelineOptionArgumentInfo, &reflection,
          &error));

  if (error) {
    return tl::unexpected(context->device->driver->status.create(
        Twine("Failed to create compute pipeline state (GPU machine code "
              "generation): ") +
        error->localizedDescription()->cString(NS::UTF8StringEncoding)));
  }

  if (!pipelineState)
    return tl::unexpected(context->device->driver->status.create(
        "Failed to create compute pipeline state (no error returned)"));

  if (!reflection)
    return tl::unexpected(context->device->driver->status.create(
        "Failed to extract pipeline argument information"));

  // Extract argument count from reflection, defaulting to 0 if unavailable
  NS::Array *arguments = reflection->arguments();
  size_t argCount = arguments ? arguments->count() : 0;

  // Create and return a MetalDeviceFunction
  return new M_driver_function(context, /*moduleName,*/ name.str(),
                               pipelineState, argCount);

  return tl::unexpected(M_DRIVER_STATUS_UNIMPLEMENTED);
}

M_driver_status_t M_driver_function::destroy(M_driver_function_t function) {
  delete function;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (function == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  auto result = M_driver_function::create(context, bundle, function_name);
  if (!result)
    return result.error();

  *function = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return M_driver_function::destroy(function);
}

M_driver_status_t
M_driver_bundle_load(M_driver_context_t context,
                     M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (runtime_bundle == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  auto result = M_driver_runtime_bundle::load(context, bundle, opts);
  if (!result)
    return result.error();

  *runtime_bundle = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_driver_runtime_bundle::unload(bundle);
}

M_driver_status_t
M_driver_runtime_bundle::unload(M_driver_runtime_bundle_t bundle) {
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  // SharedPtr releases the MTL::Library ref on destruction.
  delete bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_function_property([[maybe_unused]] M_driver_context_t context,
                           [[maybe_unused]] M_driver_function_t function,
                           const char *property_name, void *value) {
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t
M_driver_function_set_attribute([[maybe_unused]] M_driver_context_t context,
                                [[maybe_unused]] M_driver_function_t function,
                                [[maybe_unused]] int32_t attribute,
                                [[maybe_unused]] int32_t value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_function_t function,
    [[maybe_unused]] int32_t block_size,
    [[maybe_unused]] uint64_t dynamic_shared_memory_bytes,
    [[maybe_unused]] int32_t *num_blocks) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
