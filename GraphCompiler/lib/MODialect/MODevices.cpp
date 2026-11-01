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

#include "GraphCompiler/MODialect/MODevices.h"

#include "Support/AssertStream.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// The host default is built only when needed: getHostDevice uniques an
// attribute, and fusion passes call these once per kernel per scan.
DeviceRefAttr MO::getDeviceOrDefault(MOHasDeviceInterface op) {
  if (DeviceRefAttr device = op.getExecutionDevice())
    return device;
  return getHostDevice(op.getContext());
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op))
    return getDeviceOrDefault(hasDeviceOp);

  return getHostDevice(op->getContext());
}

DeviceRefAttr MO::getDeviceOrDefault(MOHasDeviceTypeInterface type) {
  if (DeviceRefAttr device = type.getDeviceRef())
    return device;
  return getHostDevice(type.getContext());
}

DeviceRefAttr MO::getDeviceOrDefaultAny(Type type) {
  if (auto hasDeviceType = dyn_cast<MOHasDeviceTypeInterface>(type))
    return getDeviceOrDefault(hasDeviceType);

  return getHostDevice(type.getContext());
}

SmallVector<DeviceRefAttr> MO::getUniqueDevicesFromKernel(Operation *op) {
  SmallVector<DeviceRefAttr> devices;
  llvm::DenseSet<DeviceRefAttr> seenDevices;

  auto addDeviceIfNew = [&](TensorType tensorType) {
    if (auto deviceRef = tensorType.getDeviceRef())
      if (seenDevices.insert(deviceRef).second)
        devices.push_back(deviceRef);
  };

  for (Value operand : op->getOperands())
    if (auto tensorType = dyn_cast<TensorType>(operand.getType()))
      addDeviceIfNew(tensorType);

  for (Value result : op->getResults())
    if (auto tensorType = dyn_cast<TensorType>(result.getType()))
      addDeviceIfNew(tensorType);

  return devices;
}

bool MO::isOnHost(Operation *op) {
  if (auto hasDeviceOp = dyn_cast<MOHasDeviceInterface>(op)) {
    auto devices = hasDeviceOp.getExecutionDevices();
    auto hostDevice = getHostDevice(op);
    return std::all_of(
        devices.begin(), devices.end(),
        [&](DeviceRefAttr device) { return device == hostDevice; });
  }
  return true;
}

bool MO::isOnHost(Type type) {
  return getDeviceOrDefaultAny(type) == getHostDevice(type.getContext());
}

//===----------------------------------------------------------------------===//
// DeviceAnnots
//===----------------------------------------------------------------------===//

static DeviceRefAttr getDeviceRef(MOHasDeviceInterface op) {
  return getDeviceOrDefault(op);
}

DeviceAnnots DeviceAnnots::forOp(Operation *op) {
  DeviceAnnots metadata;
  auto hasDevice = dyn_cast<MOHasDeviceInterface>(op);
  if (hasDevice && !op->hasTrait<MOMultiDevice>())
    metadata.device = getDeviceRef(hasDevice);
  return metadata;
}

bool DeviceAnnots::add(const DeviceAnnots &that) {
  if (!isConsistentWith(that))
    return false;
  if (!device)
    device = that.device;
  return true;
}

bool DeviceAnnots::addOp(Operation *op) {
  if (!op)
    return true;
  return add(DeviceAnnots::forOp(op));
}

Operation *DeviceAnnots::stampAny(Operation *op) const {
  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    if (device)
      hasDevice.getOperation()->setAttr(MO::kOperationDeviceRefAttrName,
                                        device);
  }
  // Ops without device support drop the annotation; AssignDevicesPass checks
  // consistency.
  return op;
}

bool DeviceAnnots::isConsistentWith(DeviceRefAttr thatDevice) const {
  return !device || !thatDevice || device == thatDevice;
}

bool DeviceAnnots::isConsistentWith(const DeviceAnnots &that) const {
  return isConsistentWith(that.device);
}

//===----------------------------------------------------------------------===//
// DeviceInfoMapping
//===----------------------------------------------------------------------===//

MO::DeviceInfoMapping::DeviceInfoMapping(mlir::ModuleOp moduleOp) {
  auto dict = moduleOp->getAttrOfType<DictionaryAttr>(kAttrName);
  if (!dict)
    return;
  for (auto entry : dict)
    if (auto infos = dyn_cast<DeviceInfoAttr>(entry.getValue()))
      _mapping[entry.getName().getValue()] = infos;
}

DeviceInfoAttr MO::DeviceInfoMapping::get(DeviceRefAttr device) const {
  auto it = _mapping.find(device.getLabel());
  if (it != _mapping.end())
    return it->second;
  return DeviceInfoAttr::get(device.getContext(), device.getLabel(), "unknown",
                             "unknown", "unknown", /*tile_based_fusion=*/false);
}
