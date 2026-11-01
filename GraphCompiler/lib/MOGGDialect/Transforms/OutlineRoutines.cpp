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
// Outlines each `mogg.routine` body into a module-level `mogg.func`
// (labeled with the `MAP` ABI) and replaces the kernel body with a single
// `mogg.call` forwarding the kernel's `!mogg.pointer` / `!mogg.dim` /
// `!mogg.context` block arguments. The `mogg.routine` container survives: it
// keeps the `!mo.tensor` interface that `MOGGToMGP` later lowers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_OUTLINEROUTINES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Outlines one `mogg.routine`: its body moves into `stub`; the kernel's body is
// rebuilt as a single `mogg.call` forwarding fresh block args to `stub`.
void outlineKernel(OpBuilder &builder, SymbolTable &symtab,
                   MOGG::RoutineOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  Location loc = kernel.getLoc();

  // The body block args (input ptrs, output ptrs, dims, context) become the
  // stub's function signature; outputs are written through pointers, so the
  // stub returns nothing.
  Block &body = kernel.getBody().front();
  SmallVector<Type> argTypes(body.getArgumentTypes());

  // A `mogg.buffer.constant` stays at the call site, where `MOGGToMGP` turns it
  // into `mgp.buffer.constant`: resources reach the runtime through the MGP-ABI
  // `void**`, which this MOGG-ABI stub lacks. Each becomes a trailing stub
  // argument now, so call sites never need patching.
  SmallVector<MOGG::BufferConstantOp> constants;
  body.walk([&](MOGG::BufferConstantOp cst) { constants.push_back(cst); });

  SmallVector<Type> stubArgTypes(argTypes);
  stubArgTypes.append(constants.size(), MOGG::PointerType::get(ctx));

  // `mogg.output`'s operands are the stub's return type; only a shape
  // function has any.
  auto bodyOutput =
      dyn_cast<MOGG::OutputOp>(body.empty() ? nullptr : &body.back());
  SmallVector<Type> stubResultTypes;
  if (bodyOutput)
    llvm::append_range(stubResultTypes, bodyOutput.getOperandTypes());

  auto functionType =
      TypeAttr::get(FunctionType::get(ctx, stubArgTypes, stubResultTypes));

  // Forward the fusion-tree provenance `MAPToMOGG` threaded onto `mogg.routine`
  // from the source `map.iter.opaque`; fall back to a fresh leaf only if it's
  // absent (e.g. a kernel region built by hand in a test).
  MOGG::FusionInfoAttr fusionInfo = kernel.getFusionInfoAttr();
  if (!fusionInfo)
    fusionInfo = MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(kernel));

  // Name the stub after the ops it fuses, e.g. "__relu_add"; the symbol table
  // appends a numeric suffix below when two kernels fuse the same ops.
  auto stubName = MOGG::getFuncName(fusionInfo);

  // Place the stub just after the kernel's top-level ancestor for readable,
  // deterministic output.
  Operation *topAncestor = kernel;
  while (topAncestor->getParentOp() &&
         !isa<ModuleOp>(topAncestor->getParentOp()))
    topAncestor = topAncestor->getParentOp();
  builder.setInsertionPointAfter(topAncestor);

  auto stub = MOGG::FuncOp::create(
      builder, loc, stubName, functionType, fusionInfo,
      /*dispatch=*/nullptr, /*constantResources=*/ArrayAttr::get(ctx, {}),
      MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MOGG),
      /*device=*/
      builder.getStringAttr(MO::getDeviceOrDefaultAny(kernel).getLabel()),
      /*trait_stub_buffer_capture=*/nullptr, /*trait_stub_arg_bounds=*/nullptr);

  // `create` already linked the stub into the module; registering it here
  // resolves any name collision and returns the unique name for the call site.
  stubName = symtab.insert(stub);

  // Move the body (block args + codegen ops) into the stub, then cap it with a
  // `mogg.output` terminator (the stub returns nothing).
  stub.getBody().takeBody(kernel.getBody());
  Block &stubBody = stub.getBody().front();
  // The constants came along with the body; rewire their uses onto the trailing
  // arguments reserved above, leaving the ops themselves unused and free to
  // move back to the call site.
  for (MOGG::BufferConstantOp cst : constants)
    cst.getPointer().replaceAllUsesWith(
        stubBody.addArgument(MOGG::PointerType::get(ctx), cst.getLoc()));
  builder.setInsertionPointToEnd(&stubBody);
  // Only a body without a terminator needs the cap.
  if (!bodyOutput)
    MOGG::OutputOp::create(builder, loc, ValueRange{});

  // Rebuild the kernel body as a single `mogg.call` forwarding fresh block
  // args (same types) to the stub, preceded by the constants it hands over.
  Block *callBody = builder.createBlock(&kernel.getBody());
  SmallVector<Value> forwarded;
  for (Type t : argTypes)
    forwarded.push_back(callBody->addArgument(t, loc));
  builder.setInsertionPointToEnd(callBody);
  for (MOGG::BufferConstantOp cst : constants) {
    cst->moveBefore(callBody, callBody->end());
    forwarded.push_back(cst.getPointer());
  }
  MOGG::CallOp::create(builder, loc, /*resultTypes=*/stubResultTypes,
                       FlatSymbolRefAttr::get(stubName), forwarded,
                       /*affinity_device_id=*/nullptr);
}

struct OutlineRoutinesPass
    : public MOGG::impl::OutlineRoutinesBase<OutlineRoutinesPass> {
  using Base::Base;

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    // Register through the cached `SymbolTableAnalysis` so names stay unique
    // and the table consistent.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    SmallVector<MOGG::RoutineOp> kernels;
    getOperation().walk([&](MOGG::RoutineOp k) {
      kernels.push_back(k);
      // No need to recurse inside a `mogg.routine`.
      return WalkResult::skip();
    });
    for (MOGG::RoutineOp kernel : kernels)
      outlineKernel(builder, symtab, kernel);
  }
};

} // namespace

} // namespace M::MOGG
