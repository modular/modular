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
//
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
// This file holds their small interfaces.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

using namespace M;
using namespace MGP;

bool DeviceGraphCacheOp::isThrows() { return true; }

// The `mgp_device_graph_alloc` primitive raises (it allocates a buffer).
bool DeviceGraphAllocOp::isThrows() { return true; }

// `recording_context()` raises (it asks the driver for a recording context).
bool DeviceGraphRecordingContextOp::isThrows() { return true; }

bool DeviceGraphDependenciesOp::producesNewNodes() {
  return !getBuilderBlockArgument().use_empty();
}

bool DeviceGraphDependenciesOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphStableCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStableCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = cast<DeviceTensorType>(adaptor.getTensor().getType());

  auto resultType = DeviceTensorType::get(
      tensorType.getDevice(), tensorType.getTensorSpec(), tensorType.getMut(),
      /*stable=*/true);

  inferredReturnTypes.push_back(resultType);
  return success();
}

bool DeviceGraphStableCreateOp::mayAlias(size_t operandIndex,
                                         size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

bool DeviceGraphStableGetOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphStablePlaceOp
//===----------------------------------------------------------------------===//

bool DeviceGraphStablePlaceOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphStableTensorOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStableTensorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceTensorType tensorType =
      cast<StableAddressType>(adaptor.getAddr().getType()).getElementType();
  inferredReturnTypes.push_back(DeviceTensorType::get(
      tensorType.getDevice(), tensorType.getTensorSpec(), tensorType.getMut(),
      /*stable=*/true));
  return success();
}

llvm::SmallBitVector DeviceGraphCollectiveCreateOp::shapeOnlyInputs() {
  llvm::SmallBitVector result(getInputs().size());

  if (std::optional<ArrayRef<int64_t>> attr = getShapeOnlyInputs()) {
    for (int64_t idx : attr.value())
      result.set(idx);
  }

  return result;
}

bool DeviceGraphCollectiveCreateOp::isThrows() { return true; }
