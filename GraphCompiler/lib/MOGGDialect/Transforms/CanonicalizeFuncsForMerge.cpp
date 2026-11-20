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

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_CANONICALIZEFUNCSFORMERGE
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Must be 0, not an out-of-range sentinel: some ops (e.g.
// `MGP::BufferConstantOp`) infer `DeviceRefAttr::getDefault` (id 0) as their
// result device and would fail re-verification otherwise. The emitted Mojo
// reads only the device label, never the id.
constexpr uint64_t kCanonicalDeviceId = 0;

struct CanonicalizeFuncsForMerge
    : public impl::CanonicalizeFuncsForMergeBase<CanonicalizeFuncsForMerge> {
  using CanonicalizeFuncsForMergeBase::CanonicalizeFuncsForMergeBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *ctx = moduleOp.getContext();

    // Device ids live on `#M.device_ref` in `!mgp.tensor` types and in a few
    // op attributes, so replace in both.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([ctx](DeviceRefAttr deviceRef) -> DeviceRefAttr {
      if (deviceRef.getId() == kCanonicalDeviceId)
        return deviceRef;
      return DeviceRefAttr::get(ctx, deviceRef.getLabel(), kCanonicalDeviceId);
    });

    // Emit reads only the rank and static dims of an op's shape attributes;
    // the runtime values ride in operands, matched by position. Dim names
    // there would keep otherwise identical kernels apart. Types are left
    // alone: block arguments and function types must keep agreeing.
    mlir::TypedAttr dyn = KGEN::ParamDeclRefAttr::get(
        mlir::StringAttr::get(ctx, "dyn"), MOSH::ShapeAttr::getDimType(ctx));
    auto canonicalShape = [&](MOSH::ShapeAttr shape) {
      auto dims =
          llvm::map_to_vector(shape.getValues(), [&](mlir::TypedAttr dim) {
            return llvm::isa<KGEN::SIMDAttr>(dim) ? dim : dyn;
          });
      return MOSH::ShapeAttr::get(dims, shape.getType());
    };

    for (FuncOp kernelStub :
         llvm::make_early_inc_range(moduleOp.getOps<FuncOp>())) {
      replacer.recursivelyReplaceElementsIn(kernelStub, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/false,
                                            /*replaceTypes=*/true);
      kernelStub.getBody().walk([&](mlir::Operation *op) {
        llvm::SmallVector<mlir::NamedAttribute> attrs(op->getAttrs());
        for (mlir::NamedAttribute attr : attrs)
          if (auto shape = llvm::dyn_cast<MOSH::ShapeAttr>(attr.getValue()))
            op->setAttr(attr.getName(), canonicalShape(shape));
      });
    }
  }
};

} // namespace

} // namespace M::MOGG
