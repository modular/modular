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
// Wraps structured MAP loops in `map.iter.opaque` regions so sequential setup
// can live outside the structured loop body.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_OUTLINESTRUCTUREDKERNELBODIES
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Wraps `loop` in a `map.iter.opaque`, moving its body into the matching
// closure. Both are isolated from above, so the opaque's block args become the
// closure's `$captures`.
void outline(StructuredIterOpInterface loop) {
  auto kernel = cast<IterOpInterface>(loop.getOperation());
  Location loc = loop.getLoc();
  OpBuilder builder(loop);

  auto opaque = IterOpaqueOp::create(builder, loc, loop->getResultTypes(),
                                     kernel.getInputs(), kernel.getInputChain(),
                                     /*fusionInfo=*/Attribute());
  // `map.iter.opaque` states its device explicitly; an unannotated loop runs
  // on the host, the `getDeviceOrDefault` fallback.
  opaque.setDeviceRef(MO::getDeviceOrDefault(
      cast<MO::MOHasDeviceInterface>(loop.getOperation())));
  if (Attribute fusionInfo = kernel.getFusionInfoAttr())
    opaque.setFusionInfoAttr(fusionInfo);
  opaque.setHasReadEffect(kernel.getHasReadEffect());
  opaque.setHasWriteEffect(kernel.getHasWriteEffect());
  opaque.setDispatch(kernel.getDispatch());

  // Chains get no block arg; the non-chain inputs and outputs are exactly the
  // closure's captures.
  Block *opaqueBody = builder.createBlock(&opaque.getBody());
  SmallVector<Value> bodyCaptures;
  for (Value input : kernel.getInputs())
    bodyCaptures.push_back(opaqueBody->addArgument(input.getType(), loc));
  for (Type resultType : loop->getResultTypes())
    if (!isa<MO::ChainType>(resultType))
      bodyCaptures.push_back(opaqueBody->addArgument(resultType, loc));

  builder.setInsertionPointToStart(opaqueBody);
  CapturingFusionClosure body =
      loop.createBodyClosure(builder, loc, bodyCaptures);

  // Move the body without cloning. The loop's block args are [inputs...,
  // outputs..., index] and the closure wants [index, captures...], so only
  // the index moves.
  BlockArgument oldIndexArg = loop.getIndexArg();
  Region &region = body.getBody();
  region.takeBody(loop->getRegion(0));
  Block &movedBody = region.front();
  BlockArgument newIndexArg =
      movedBody.insertArgument(0u, oldIndexArg.getType(), loc);
  oldIndexArg.replaceAllUsesWith(newIndexArg);
  movedBody.eraseArgument(movedBody.getNumArguments() - 1);

  loop->replaceAllUsesWith(opaque.getResults());
  loop->erase();
}

struct OutlineStructuredKernelBodiesPass
    : public MAP::impl::OutlineStructuredKernelBodiesBase<
          OutlineStructuredKernelBodiesPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<StructuredIterOpInterface> worklist;
    getOperation()->walk([&](StructuredIterOpInterface op) {
      worklist.push_back(op);
      return WalkResult::skip();
    });

    for (StructuredIterOpInterface op : worklist)
      outline(op);
  }
};

} // namespace

} // namespace M::MAP
