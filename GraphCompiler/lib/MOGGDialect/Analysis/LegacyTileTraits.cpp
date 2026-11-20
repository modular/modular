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

#include "GraphCompiler/MOGGDialect/Analysis/LegacyTileTraits.h"
#include "GraphCompiler/MOGGDialect/Support/PlatformPredicate.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::MOGG;

namespace {

/// One registration allowed to bind a legacy tile trait: the op name and the
/// platform of its registration, as `PlatformPredicate::str` spells it, or
/// "true" for `register(name)`.
struct LegacyTileRegistration {
  llvm::StringLiteral registration;
  llvm::StringLiteral cond = "true";

  bool matches(llvm::StringRef name, const PlatformPredicate &platform) const {
    return name == registration && platform.str() == cond;
  }
};

/// The only registrations that may bind a legacy tile trait.
///
/// DO NOT ADD TO THIS LIST. It holds the kernels that used the tile IOSpecs
/// when the traits replaced them, and it only shrinks. A new tile kernel waits
/// for the tile epilogue interface of GEX-4467.
constexpr LegacyTileRegistration kLegacyTileRegistrations[] = {
    {"mo.matmul", R"(type("gpu") & api("neuron"))"},
    {"mo.composite.reduce.rms_norm", R"(type("gpu") & api("neuron"))"},
    {"mo.composite.rms_norm_fused_residual_add",
     R"(type("gpu") & api("neuron"))"},
    // The sm_100a tile kernels: `MatmulTileSM100` and the `tile_rms_norm`
    // test kernels.
    {"mo.matmul", R"(type("gpu") & api("cuda") & arch("sm_100a"))"},
    {"tile_rms_norm", R"(type("gpu") & api("cuda") & arch("sm_100a"))"},
};

bool bindsLegacyTileTrait(KernelFunctionAttr fn) {
  if (!fn)
    return false;
  return llvm::any_of(fn.getTensorArgContracts(), [](mlir::Attribute attr) {
    std::optional<TensorArgContract> contract = dictToTensorArgContract(attr);
    return contract && legacyTileLane(*contract).has_value();
  });
}

} // namespace

std::optional<LegacyTileLane>
M::MOGG::legacyTileLane(const TensorArgContract &contract) {
  if (contract.hasAccess(TensorArgAccess::legacyTileStore))
    return LegacyTileLane::store;
  if (contract.hasAccess(TensorArgAccess::legacyTileTransform))
    return LegacyTileLane::compute;
  return std::nullopt;
}

bool M::MOGG::isLegacyTileFusedOutput(const KernelDeclOperandAdaptor &arg) {
  std::optional<TensorArgContract> contract = arg.getTraitContract();
  return contract && contract->fused &&
         contract->role == TensorArgRole::output &&
         legacyTileLane(*contract).has_value();
}

mlir::LogicalResult M::MOGG::verifyLegacyTileRegistration(
    mlir::Location loc, llvm::StringRef registration,
    const PlatformPredicate &platform,
    llvm::ArrayRef<KernelFunctionAttr> functions) {
  if (llvm::none_of(functions, bindsLegacyTileTrait))
    return mlir::success();
  if (llvm::any_of(kLegacyTileRegistrations,
                   [&](const LegacyTileRegistration &allowed) {
                     return allowed.matches(registration, platform);
                   }))
    return mlir::success();

  std::string cond = platform.str();
  return mlir::emitError(loc)
         << "kernel '" << registration << "'"
         << (cond == "true" ? std::string() : " (cond=" + cond + ")")
         << " binds a legacy tile trait (_LegacyTileStoreAccess or "
            "_LegacyTileComputeOutput); only the registrations listed in "
            "LegacyTileTraits.cpp may use them";
}
