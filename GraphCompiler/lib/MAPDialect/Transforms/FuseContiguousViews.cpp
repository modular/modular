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
// Fuses each `mo.static.reshape` into the `map.iter.opaque` kernels that read
// it, as a zero-copy `map.index.reshape.stride` view in the kernel's body.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MOOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_FUSECONTIGUOUSVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Rewrites each input of `kernel` that a reshape produces to read the
// reshape's input through a stride view at the top of the body. The view keeps
// the reshape's result type, so the call operands and closure captures that
// read the old input stay well typed. The old input and the reshape are left
// to the canonicalizer.
unsigned fuseReshapesInto(IterOpaqueOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  OpBuilder builder = OpBuilder::atBlockBegin(&kernel.getBody().front());

  llvm::DenseMap<Operation *, Value> views;
  unsigned numFused = 0;
  // The reshapes' inputs are appended, so the loop never visits one.
  for (unsigned idx = 0, e = kernel.getInputs().size(); idx < e; ++idx) {
    auto reshape =
        kernel.getInputs()[idx].getDefiningOp<MO::Static::ReshapeOp>();
    if (!reshape)
      continue;

    Value &view = views[reshape];
    if (!view) {
      Value source = reshape.getInput();
      kernel.insertInputsAt(kernel.getInputs().size(), source);
      auto viewTy = cast<MO::TensorType>(reshape.getResult().getType());
      auto spec = ReshapeSpecAttr::get(
          ctx, ViewSignatureAttr::get(
                   ctx, cast<MO::TensorType>(source.getType()).getShape(),
                   viewTy.getShape()));
      view = ReshapeStrideOp::create(builder, reshape.getLoc(), viewTy, spec,
                                     kernel.getInputBlockArgs().back());
    }
    kernel.getInputBlockArgs()[idx].replaceAllUsesWith(view);
    ++numFused;
  }
  return numFused;
}

struct FuseContiguousViewsPass
    : public MAP::impl::FuseContiguousViewsBase<FuseContiguousViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Walk, since a kernel can sit inside control flow (e.g. mo.while).
    SmallVector<IterOpaqueOp> kernels;
    getOperation().walk([&](IterOpaqueOp op) { kernels.push_back(op); });

    unsigned fused = 0;
    for (IterOpaqueOp kernel : kernels)
      fused += fuseReshapesInto(kernel);
    numViewsFused = fused;
  }
};

} // namespace

} // namespace M::MAP
