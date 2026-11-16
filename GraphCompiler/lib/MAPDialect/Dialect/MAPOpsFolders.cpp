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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// The declarative formats name these unqualified.
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

using namespace M;
using namespace mlir;
using namespace M::MAP;
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

OpFoldResult ConstantScalarOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// map.constant
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValueAttr(); }
