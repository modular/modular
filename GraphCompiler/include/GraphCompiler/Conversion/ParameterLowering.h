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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H

#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Value.h"

namespace M {

class ParameterLoweringState {
public:
  using PlaceholderOp = MOSH::ParamToValueOp;

  /// Callback function implementing the specific lowering for expression
  /// attributes.
  using LowerExprFunc = llvm::function_ref<ErrorOr<mlir::Value>(
      ParameterLoweringState &, OpBuilder &, Location, Region *, TypedAttr,
      ValueRange)>;

  /// Register what SSA value the parameter attribute is lowered to. Prefer to
  /// use the higher level lowering methods where placeholders are concerned.
  void emplaceLowered(Region *parentRegion, Attribute attr, Value value) {
    loweredParams[parentRegion].try_emplace(attr, value);
  }

  /// Returns the SSA value that the parameter attribute is lowered to, or a
  /// null value if it hasn't been lowered yet.
  Value lookupLowered(Region *parentRegion, Attribute attr) {
    while (parentRegion) {
      auto &regionMap = loweredParams[parentRegion];
      if (auto it = regionMap.find(attr); it != regionMap.end())
        return it->second;
      parentRegion = parentRegion->getParentRegion();
    }

    return {};
  }

  void eraseLowered(Region *parentRegion, Attribute attr) {
    while (parentRegion) {
      auto &regionMap = loweredParams[parentRegion];
      if (regionMap.erase(attr))
        return;
      parentRegion = parentRegion->getParentRegion();
    }

    assert(false && "unable to find parent region for attribute");
    return;
  }

  /// Returns a lowered value for the parameter expression attribute. Parameter
  /// references not seen before are lowered to a placeholder, parameter
  /// operators are lowered recursively, and all other attributes are delegated
  /// to the provided hook (using the given builder and location). The hook
  /// should not update the value mapping.
  ErrorOr<Value> lowerExpression(OpBuilder &b, Location loc,
                                 Region *parentRegion, TypedAttr attr,
                                 LowerExprFunc func);

  /// Lower a parameter declaration, resolving the placeholder if necessary. If
  /// a placeholder is resolved, its SSA value is returned so that callers can
  /// replace their uses if necessary. Otherwise, a null Value signifies a
  /// successful lowering without placeholder resolution.
  Value lowerParamDecl(Region *parentRegion, KGEN::ParamDeclAttr declAttr,
                       Value value);

private:
  /// Helper to lower an array of expression operands.
  ErrorOr<SmallVector<Value>> lowerOperands(OpBuilder &b, Location loc,
                                            Region *parentRegion,
                                            ArrayRef<TypedAttr> operands,
                                            LowerExprFunc func);

  /// Tracks what parameters attributes (e.g. expressions) are lowered to.
  ///  - Dimension parameter attributes are lowered to values of type `index`.
  ///  - Shape parameter attributes are lowered to values of type
  ///    `mgp.tensor_spec<...xunk>`.
  DenseMap<Region *, DenseMap<Attribute, Value>> loweredParams;
};

} // namespace M

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_PARAMETERLOWERING_H
