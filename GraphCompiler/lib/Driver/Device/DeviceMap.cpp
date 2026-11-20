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

#include "GraphCompiler/Driver/Device/DeviceMap.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace M::GraphCompiler::Driver;
using namespace M;

ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
M::GraphCompiler::Driver::mapDeviceRefsToDevices(
    const ArrayRef<DeviceRef> deviceRefs, const DeviceMap &deviceMap,
    CPUDevice &cpuDevice) {

  // Until MGP shares the driver's DeviceContext, sync so no driver alloc/copy
  // is pending when the first kernel touches the resource.
  for (auto &dev : deviceMap) {
    if (dev.second) {
      if (auto err = dev.second->synchronize())
        return err.takeError();
    }
  }

  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> deviceContexts;
  for (const auto &devRef : deviceRefs) {
    // CPU device not explicitly set so deviceMap["cpu"] may be null
    if (devRef.label == kCPULabel) {
      ErrorOr<M::Driver::DeviceContextRef> ctxOr =
          M::Driver::DeviceContext::create("cpu", 0);
      if (ctxOr.isError())
        return ctxOr.takeError();
      deviceContexts.push_back(
          AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
              cpuDevice, ctxOr.takeValue()));
    } else {
      if (auto it = deviceMap.find(devRef);
          it != deviceMap.end() && it->second) {
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, it->second));
      } else {
        ASSERT_STREAM(devRef.label == kGPULabel,
                      "Only accelerator device context supported");
        // Uses the device ref id as the device id, unlike when the device is
        // passed to the graph.
        // TODO: Move device selection into the mt command and drop this branch.
        size_t numPhysicalDevices =
            M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);
        if (devRef.id >= numPhysicalDevices)
          return Error(devRef.toString() +
                       " id exceeds number of available accelerator devices.");
        ErrorOr<M::Driver::DeviceContextRef> ctxOr =
            M::Driver::DeviceContext::create(M::Driver::Device::gpuAPI,
                                             devRef.id);
        if (ctxOr.isError())
          return ctxOr.takeError();
        deviceContexts.push_back(
            AnyAsyncValueRef::createReady<M::Driver::DeviceContextRef>(
                cpuDevice, ctxOr.takeValue()));
      }
    }
  }
  return deviceContexts;
}
