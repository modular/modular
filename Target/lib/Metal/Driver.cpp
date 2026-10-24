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

#include "Driver.h"
#include "Common.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

M_driver_driver_t M_driver_driver::create() {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto driver = std::make_unique<M_driver_driver>();
  if (!driver)
    return nullptr;

  NS::SharedPtr<NS::Array> mDevices = NS::TransferPtr(MTL::CopyAllDevices());
  if (!mDevices)
    return nullptr;

  driver->devices.reserve(mDevices->count());
  for (NS::UInteger i = 0; i < mDevices->count(); i++) {
    driver->devices.emplace_back(
        driver.get(), static_cast<MTL::Device *>(mDevices->object(i)));
  }

  return driver.release();
}

void M_driver_driver::destroy(M_driver_driver *driver) { delete driver; }

M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *outDriver) {
  if (version == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (outDriver == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (version->major != M_DRIVER_INTERFACE_VERSION_MAJOR)
    return M_DRIVER_STATUS_INCOMPATIBLE_VERSION;

  version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
  version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
  version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;

  M_driver_driver *driver = M_driver_driver::create();
  if (!driver)
    return M_DRIVER_STATUS_NO_DEVICES;
  *outDriver = driver;
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  M_driver_driver::destroy(handle);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                  const char *property_name, void *value) {
  StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API)
    *static_cast<const char **>(value) = "Metal";
  else if (name == M_DRIVER_PROPERTY_VERSION) {
    M_driver_version *version = static_cast<M_driver_version *>(value);
    version->major = M_DRIVER_INTERFACE_VERSION_MAJOR;
    version->minor = M_DRIVER_INTERFACE_VERSION_MINOR;
    version->patch = M_DRIVER_INTERFACE_VERSION_PATCH;
  } else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
