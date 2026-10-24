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
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H

#include "GraphCompiler/Interfaces/EmitsMojoInterface.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Needed because ODS-generated build methods for mogg.kernel.decl use the
// unqualified name.
using M::KGEN::LIT::FnTypeGeneratorType;

namespace M::MOGG {
template <typename ConcreteType>
class CallLike : public mlir::OpTrait::TraitBase<ConcreteType, CallLike> {};
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MOGGDialect/MOGG.h.inc"

namespace M::MOGG {

/// Returns `__<ops>`, a stub name built from the ops `fusionInfo` fuses. The
/// caller inserts the stub into a symbol table for a unique name.
StringAttr getFuncName(FusionInfoAttr fusionInfo);

template <typename Kernel>
SmallVector<KGEN::ParamDeclRefAttr> getKernelUsedParameters(Kernel kernel) {
  DenseSet<KGEN::ParamDeclRefAttr> seenParams;
  SmallVector<KGEN::ParamDeclRefAttr> usedParameters;

  auto params = kernel.getOutputParamDecls();
  auto isInOutputParams = [&](StringRef name) {
    for (KGEN::ParamDeclAttr attr : params) {
      if (attr.getName().str() == name)
        return true;
    }
    return false;
  };

  // Crawls parameters nested in expression attributes.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](KGEN::ParamDeclRefAttr param) {
    if (auto [_, inserted] = seenParams.insert(param); !inserted)
      return WalkResult::advance();

    if (isInOutputParams(param.getName()))
      return WalkResult::advance();

    usedParameters.push_back(param);
    return WalkResult::advance();
  });

  auto addParamsFromTensor = [&](Type t) {
    MOSH::ShapeAttr shape;

    MO::TensorType asTensor = MO::dynCastFromTensorLike(t);
    if (asTensor)
      shape = asTensor.getShape();

    if (!shape)
      return;
    walker.walk(shape);
  };

  auto addParamsFromOp = [&](Operation *op) {
    for (Type t : op->getOperandTypes())
      addParamsFromTensor(t);

    for (Type t : op->getResultTypes())
      addParamsFromTensor(t);

    if (auto getDim = dyn_cast<MOGG::GetDimOp>(op)) {
      auto ref = KGEN::ParamDeclRefAttr::get(
          getDim.getDimension(), MOSH::ShapeAttr::getDimType(op->getContext()));
      auto itr = seenParams.find(ref);

      if (itr == seenParams.end()) {
        usedParameters.push_back(ref);
        seenParams.insert(ref);
      }
    }
  };

  addParamsFromOp(kernel);

  // Rarely, a kernel uses a parameter internally that is on neither its inputs
  // nor outputs, e.g. `J` here:

  // clang-format off
  // %0 = broadcast(%input) : tensor<[N]> -> tensor<N, J>
  // %1 = broadcast(%0) : tensor<[N, J]> -> tensor<N, B>
  // %3 = broadcast(%1) : tensor<[N, B, K]> -> tensor<N, B, K, M>
  // clang-format on

  kernel.walk(addParamsFromOp);

  return usedParameters;
}
} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
