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

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GraphCompiler/MOGGDialect/Support/MojoTraits.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPOps.h"

using namespace M;

using namespace M;

namespace M::MOGG {

std::optional<StringRef>
EmitMojoState::getResultParamNameFor(const KernelDeclFunctionAdaptor &fnAdaptor,
                                     StringRef paramName) {
  if (!fnAdaptor.attr.getResultParams())
    return std::nullopt;
  // One entry per result: a {struct-param-name: value} DictionaryAttr if
  // ImportPackages saw the struct's declaration, else a UnitAttr. Only an
  // unresolved value (a ParamDeclRefAttr) has a name to return.
  for (auto resultParam : fnAdaptor.attr.getResultParams()) {
    auto resultParamDict = dyn_cast<DictionaryAttr>(resultParam);
    if (!resultParamDict)
      continue;
    auto attr = resultParamDict.get(paramName);
    if (auto funcParamName = dyn_cast_or_null<KGEN::ParamDeclRefAttr>(attr))
      return funcParamName.getName();
  }
  return std::nullopt;
}

// Returns the trace name from fusionInfo, or empty string if not set.
static MojoValue getTraceNameValue(FusionInfoAttr fusionInfo) {
  // GEX-979: pass in trace description for fused kernel without
  // invalidating cache.
  if (!fusionInfo)
    return MojoValue("").quote();
  return MojoValue(fusionInfo.toString()).quote();
}

static MojoValue getHasEpilogueFusionValue(mlir::Operation *) {
  // Fused epilogues set `CallOpaqueOp::getHasEpilogueFusion`, which the call's
  // `emitMojo` honors; a kernel without that flag has none.
  return false;
}

ErrorOr<MojoValue> EmitMojoState::inferDefaultKernelParamValue(
    Operation *callOp, StringRef paramName, DictionaryAttr customParamDict) {

  auto inferFromMlirParam = [&](Attribute customParam) -> ErrorOr<MojoValue> {
    // TODO: This conversion is hardcoded and brittle; it needs the parameter
    // type from the kernel infos, not just the name.
    if (auto strAttr = dyn_cast<StringAttr>(customParam))
      return MojoValue(strAttr.getValue()).quote();
    else if (auto boolAttr = dyn_cast<BoolAttr>(customParam))
      return boolAttr.getValue();
    else if (auto intAttr = dyn_cast<IntegerAttr>(customParam))
      // getInt() asserts signless or index; this may be signed (e.g. si32).
      return intAttr.getValue().getSExtValue();
    else if (auto elems = dyn_cast<ElementsAttr>(customParam);
             elems && elems.getShapedType().getRank() == 0 &&
             isa<FloatType>(elems.getElementType())) {
      // A float scalar param is a rank-0 elements attribute (see
      // MO_F32ScalarAttr). `#M.dense_array` only iterates as `Attribute`.
      auto values = elems.tryGetValues<Attribute>();
      auto scalar = values ? dyn_cast<FloatAttr>(*values->begin()) : nullptr;
      if (!scalar || !scalar.getValue().isFinite())
        return Error(
            llvm::formatv("Float param `{}` has no Mojo literal", paramName));
      return MojoValue(scalar.getValue());
    } else if (auto dtypeAttr = dyn_cast<KGEN::DTypeConstantAttr>(customParam))
      return EmitMojoState::mlirDTypeToMojoType(dtypeAttr.getDType());
    else if (auto arrAttr = dyn_cast<ArrayAttr>(customParam);
             arrAttr && !arrAttr.empty() && isa<IntegerAttr>(arrAttr[0])) {
      SmallVector<int64_t> values;
      for (auto val : arrAttr)
        values.push_back(cast<IntegerAttr>(val).getValue().getSExtValue());
      return EmitMojoState::getAsCoordMojoValue(values);
    } else
      return Error(llvm::formatv("Unsupported custom param MLIR type for `{}`",
                                 paramName));
  };

  if (auto kernelCallOp = dyn_cast<KernelCallInterface>(callOp);
      kernelCallOp && kernelCallOp.getKernelParams()) {
    if (auto customParam = kernelCallOp.getKernelParams().get(paramName))
      return inferFromMlirParam(customParam);
  }

  if (customParamDict) {
    if (auto customParam = customParamDict.get(paramName))
      return inferFromMlirParam(customParam);
  }

  // Some special parameters are common to all kernels.
  // TODO(GEX-2847): Check the op, not the FuncOp, in case these get
  // sequence fused. Safe today: stubs are outlined before sequence fusion, and
  // sequence-fused ones are all on the host.
  if (paramName == MOGG::kMoggTraceNameParameterName) {
    auto kernelStubOp = callOp->getParentOfType<FuncOp>();
    return getTraceNameValue(kernelStubOp.getFusionInfoAttr());
  }
  if (paramName == MOGG::kMoggHasEpilogueFusionParameterName)
    return getHasEpilogueFusionValue(callOp);

  return Error(llvm::formatv("Failed to infer parameter `{}`", paramName));
}
} // namespace M::MOGG
