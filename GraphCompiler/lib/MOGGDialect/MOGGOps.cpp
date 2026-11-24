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
// This file implements the MOGG dialect operations.
//
//===----------------------------------------------------------------------===//
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include <vector>

using namespace M;
using namespace MOGG;

StringAttr M::MOGG::getFuncName(FusionInfoAttr fusionInfo) {
  ASSERT_STREAM(fusionInfo, << "cannot name a kernel stub without fusion info");
  // The hash keeps names that share a truncated prefix distinct in Mojo.
  return StringAttr::get(
      fusionInfo.getContext(),
      "__" + joinNameParts(fusionInfo.getLeafNames(), TruncationMarker::Hash));
}

//===----------------------------------------------------------------------===//
// KernelDeclOp
//===----------------------------------------------------------------------===//

/// Returns the location named `name` among `declLoc`'s fused locations.
static std::optional<Location> findNamedLoc(Location declLoc, StringRef name) {
  auto fused = dyn_cast<mlir::FusedLoc>(declLoc);
  if (!fused)
    return std::nullopt;
  for (Location loc : fused.getLocations())
    if (auto named = dyn_cast<mlir::NameLoc>(loc);
        named && named.getName() == name)
      return named.getChildLoc();
  return std::nullopt;
}

Location KernelDeclOp::getMojoStructLocation() {
  return findNamedLoc(getLoc(), kStructLocationName).value_or(getLoc());
}

Location KernelDeclOp::getMojoFunctionLocation(KernelFunctionKind kind) {
  if (std::optional<Location> loc =
          findNamedLoc(getLoc(), stringifyKernelFunctionKind(kind)))
    return *loc;
  return getMojoStructLocation();
}

std::string KernelDeclOp::getMojoArgLocationName(KernelFunctionKind kind,
                                                 uint64_t position) {
  return llvm::formatv("{0}.arg{1}", stringifyKernelFunctionKind(kind),
                       position);
}

Location KernelDeclOp::getMojoArgLocation(KernelFunctionKind kind,
                                          uint64_t position) {
  std::string name = getMojoArgLocationName(kind, position);
  if (std::optional<Location> loc = findNamedLoc(getLoc(), name))
    return *loc;
  return getMojoFunctionLocation(kind);
}

//===----------------------------------------------------------------------===//
// GraphBlobOp
//===----------------------------------------------------------------------===//

void GraphBlobOp::build(OpBuilder &builder, OperationState &state,
                        function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(state.addRegion());
  bodyBuilder(builder, state.location);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOGGDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MOGGDialect/MOGG.cpp.inc"
      >();
}
