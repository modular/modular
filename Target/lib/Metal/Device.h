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

#ifndef TARGET_LIB_METAL_DEVICE_H
#define TARGET_LIB_METAL_DEVICE_H

#include "Common.h"
#include "MetalCpp.h"
#include "llvm/ADT/StringRef.h"

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver, MTL::Device *mDevice)
      : driver(driver), mDevice(mDevice) {}

  DriverResult<int> getComputeCapability(llvm::StringRef name);
  M_driver_status_t getProperty(llvm::StringRef name, void *value);

  M_driver_driver_t driver;
  MTL::Device *mDevice;
};

#endif // TARGET_LIB_METAL_DEVICE_H
