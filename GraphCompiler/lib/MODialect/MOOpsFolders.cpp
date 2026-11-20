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
// This file implements folding and canonicalization for the MO dialect
// operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

using namespace M;
using namespace MO;
//===----------------------------------------------------------------------===//
// ReduceAddOp, ReduceMinOp, ReduceMaxOp, ReduceMulOp

// A reduction that changes no dimension returns its input.
static OpFoldResult foldImplicitAxisReduction(TensorValue input,
                                              TensorValue result) {
  if (input.getType() == result.getType())
    return input;
  return {};
}

OpFoldResult ReduceAddOp::fold(FoldAdaptor) {
  return foldImplicitAxisReduction(getInput(), getResult());
}

OpFoldResult ReduceMinOp::fold(FoldAdaptor) {
  return foldImplicitAxisReduction(getInput(), getResult());
}

OpFoldResult ReduceMaxOp::fold(FoldAdaptor) {
  return foldImplicitAxisReduction(getInput(), getResult());
}

OpFoldResult ReduceMulOp::fold(FoldAdaptor) {
  return foldImplicitAxisReduction(getInput(), getResult());
}
