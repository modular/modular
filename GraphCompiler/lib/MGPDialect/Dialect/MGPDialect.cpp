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

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FoldInterfaces.h"

using namespace M;
using namespace MGP;

namespace {

//===----------------------------------------------------------------------===//
// MGPDialectFoldInterface
//===----------------------------------------------------------------------===//

struct MGPDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Keep constants inside a `mgp.device_graph.collective` body. Each region
  /// records a separate per-device graph, so a constant hoisted above the op
  /// would turn into a value captured from the enclosing scope and shared by
  /// every device's recording instead of staying local to the graph using it.
  bool shouldMaterializeInto(Region *region) const override {
    return isa<DeviceGraphCollectiveOp>(region->getParentOp());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// MGPDialect
//===----------------------------------------------------------------------===//

void MGPDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
  addInterfaces<MGPTensorBufferDialectInterface, MGPDialectFoldInterface>();

  // PrepareDevicePrimitivesPass appends the device label to device-generic op
  // names for GraphRT primitive dispatch; those ops are not registered.
  allowUnknownOperations();
}

Operation *MGPDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value))
    return mlir::index::ConstantOp::create(builder, loc, type, intAttr);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.cpp.inc"
