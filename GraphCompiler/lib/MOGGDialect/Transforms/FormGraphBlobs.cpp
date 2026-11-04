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
// Fuses groups of declarative `mogg.routine`s, sequences and siblings alike,
// into one routine whose body is a `mogg.graph_blob`, so that each group
// launches as one device kernel. The fused routine has every member's operands
// and results, and `OutlineRoutines` outlines it like any other.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FORMGRAPHBLOBS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Whether `routine` can join a blob: its one call is to a declarative kernel,
// on tensors, named dims and one device context, with no chain and no result
// with a data-dependent shape, and some output isn't in `invariant`. A routine
// computing only invariant values runs once at init, and a blob runs on every
// execution. `symbols` holds the kernel decls.
bool canJoinBlob(RoutineOp routine, SymbolTable &symbols,
                 const llvm::DenseSet<Value> &invariant) {
  // No chain, and no result with a data-dependent shape.
  if (routine.getChain() || !routine.getOutputParamDecls().empty())
    return false;

  // Some output varies, so a routine with no outputs stays out too.
  if (llvm::all_of(routine.getOutputs(),
                   [&](Value value) { return invariant.contains(value); }))
    return false;

  // All inputs and outputs are tensors. A buffer, which a kernel updates in
  // place, waits for declarative kernels to take mutable inputs (GEX-4429).
  auto isTensor = [](Type type) { return isa<MO::TensorType>(type); };
  if (!llvm::all_of(routine.getInputs().getTypes(), isTensor) ||
      !llvm::all_of(routine.getOutputs().getTypes(), isTensor))
    return false;

  // Only named dims, and one context. The blob shares one argument per dim
  // name, so an unnamed dim could stand for any dim.
  RoutineOp::BodyArgs args = routine.getBodyArgs();
  auto isNamedDim = [](BlockArgument arg) {
    return static_cast<bool>(cast<DimType>(arg.getType()).getName());
  };
  if (!llvm::all_of(args.dims, isNamedDim) || args.contexts.size() != 1)
    return false;

  // Exactly one call, to a declarative kernel on one device context, which the
  // blob makes a `mogg.call.execute_device`.
  CallOpaqueOp call;
  for (Operation &op : routine.getBody().front()) {
    if (!op.hasTrait<CallLike>())
      continue;
    if (call)
      return false;
    call = dyn_cast<CallOpaqueOp>(op);
    if (!call)
      return false;
  }
  if (!call || call.getDeviceContexts().size() != 1)
    return false;
  auto decl = symbols.lookup<KernelDeclOp>(call.getCallee());
  return decl && decl.getExecuteDevice();
}

// Replaces `members` with one routine whose body is a `mogg.graph_blob` holding
// their bodies in order. It goes where the last member was, after every input
// of the blob. Nothing outside the blob reads a member, so every value is still
// defined before its uses.
void fuse(ArrayRef<RoutineOp> members) {
  RoutineOp last = members.back();
  MLIRContext *ctx = last.getContext();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(last);
  SmallVector<Location> locs;
  for (RoutineOp member : members)
    locs.push_back(member.getLoc());
  Location loc = rewriter.getFusedLoc(locs);

  // The blob reads what its members read from outside it, and writes every
  // member's results: `MOGGToMGP` gives each its own buffer.
  SmallVector<Value> inputs;
  llvm::DenseMap<Value, unsigned> inputIndex;
  SmallVector<Value> results;
  llvm::DenseMap<Value, unsigned> resultIndex;
  for (RoutineOp member : members) {
    for (Value input : member.getInputs()) {
      if (!resultIndex.contains(input) &&
          inputIndex.try_emplace(input, inputs.size()).second)
        inputs.push_back(input);
    }
    for (Value result : member.getOutputs()) {
      resultIndex[result] = results.size();
      results.push_back(result);
    }
  }

  SmallVector<Operation *> memberOps(members.begin(), members.end());
  auto fused = RoutineOp::create(
      rewriter, loc, ValueRange(results).getTypes(), inputs, /*chain=*/Value(),
      createFusionInfo<FusionSequenceAttr>(ctx, memberOps));
  fused.setDeviceRef(MO::getDeviceOrDefaultAny(last));
  bool deferred = llvm::any_of(members, [](RoutineOp member) {
    return member.getDispatch() == DispatchKind::Deferred;
  });
  fused.setDispatch(deferred ? DispatchKind::Deferred
                             : DispatchKind::Immediate);

  Block *body = rewriter.createBlock(&fused.getBody());
  for (size_t i = 0, e = inputs.size() + results.size(); i < e; ++i)
    body->addArgument(PointerType::get(ctx), loc);
  // A dim is named by its type, so members share one argument per name.
  llvm::DenseMap<Type, BlockArgument> dims;
  for (RoutineOp member : members) {
    for (BlockArgument dim : member.getBodyArgs().dims) {
      if (!dims.contains(dim.getType()))
        dims[dim.getType()] = body->addArgument(dim.getType(), loc);
    }
  }
  BlockArgument context = body->addArgument(ContextType::get(ctx), loc);
  auto resultPointer = [&](Value result) {
    return body->getArgument(inputs.size() + resultIndex.at(result));
  };

  GraphBlobOp::create(rewriter, loc, [&](OpBuilder &builder, Location) {
    for (RoutineOp member : members) {
      RoutineOp::BodyArgs args = member.getBodyArgs();
      IRMapping mapping;
      for (auto [input, arg] :
           llvm::zip_equal(member.getInputs(), args.inputs)) {
        mapping.map(arg, resultIndex.contains(input)
                             ? resultPointer(input)
                             : body->getArgument(inputIndex.at(input)));
      }
      for (auto [result, arg] :
           llvm::zip_equal(member.getOutputs(), args.outputs))
        mapping.map(arg, resultPointer(result));
      for (BlockArgument dim : args.dims)
        mapping.map(dim, dims.at(dim.getType()));
      mapping.map(args.contexts.front(), context);
      auto mapped = [&](ValueRange values) {
        return llvm::map_to_vector(
            values, [&](Value value) { return mapping.lookup(value); });
      };
      for (Operation &op : member.getBody().front()) {
        // In the blob, the member's call runs its kernel's `execute_device`.
        if (auto call = dyn_cast<CallOpaqueOp>(op))
          CallExecuteDeviceOp::create(
              builder, call.getLoc(), call.getCalleeAttr(),
              mapped(call.getInputs()), mapped(call.getOutputs()),
              mapping.lookup(call.getDeviceContexts().front()),
              call.getKernelParamAttr());
        else
          builder.clone(op, mapping);
      }
    }
  });

  rewriter.replaceAllUsesWith(results, fused.getOutputs());
  for (RoutineOp member : llvm::reverse(members))
    rewriter.eraseOp(member);
}

// The values `op` reads: its operands, and the values from above that its
// regions read, such as an elementwise loop's inputs.
llvm::SetVector<Value> readValues(Operation &op) {
  llvm::SetVector<Value> values(op.operand_begin(), op.operand_end());
  getUsedValuesDefinedAbove(op.getRegions(), values);
  return values;
}

// The values of a graph's body that are the same on every execution, so they
// can be computed once at init, as MGP's `ExecInvariantCodeMotion` does.
// `body` must be in dataflow order.
llvm::DenseSet<Value> findInvariantValues(Block &body) {
  llvm::DenseSet<Value> invariant;
  auto isInvariant = [&](Value value) { return invariant.contains(value); };
  auto isEffect = [](Type type) {
    return isa<MO::ChainType, MO::BufferType>(type);
  };
  for (Operation &op : body) {
    // A constant is invariant, unless it's an external one something else can
    // update between executions.
    if (auto constant = dyn_cast<MO::ConstantLike>(op)) {
      auto external = dyn_cast<MO::ConstantExternalOp>(op);
      if (!external || !external.getHasAlias())
        invariant.insert(constant.getResult());
      continue;
    }
    // TODO(GEX-4484): Find the invariant values of a `mo.parallel` body, which
    // `MOGGToMGP` stamps out once per device. Until then it counts as variant,
    // like control flow.
    if (isa<RegionBranchOpInterface, MO::ParallelOp>(op) ||
        llvm::any_of(op.getOperandTypes(), isEffect) ||
        llvm::any_of(op.getResultTypes(), isEffect) ||
        !llvm::all_of(readValues(op), isInvariant))
      continue;
    invariant.insert(op.result_begin(), op.result_end());
  }
  return invariant;
}

// Whether anything but a routine that can join `member`'s blob, or the block's
// terminator, reads one of its results.
bool isReadOutside(RoutineOp member, SymbolTable &symbols,
                   const llvm::DenseSet<Value> &invariant) {
  Block *block = member->getBlock();
  auto device = MO::getDeviceOrDefaultAny(member);
  return llvm::any_of(member->getUsers(), [&](Operation *user) {
    // A reader nested in a region, such as a `mo.if` branch, is outside: only
    // routines directly in `block` join its blobs.
    if (user->getBlock() != block)
      return true;
    if (user->hasTrait<OpTrait::IsTerminator>())
      return false;
    auto routine = dyn_cast<RoutineOp>(user);
    return !routine || !canJoinBlob(routine, symbols, invariant) ||
           MO::getDeviceOrDefaultAny(routine) != device;
  });
}

// Forms blobs in `block`, which is in dataflow order, keeping one blob open.
//
// Nothing outside the open blob reads its values: the blob ends at the first
// member with a result that something outside it needs. That result is ready
// as soon as the blob ends, rather than waiting for later members it doesn't
// depend on, and no path can leave a blob and come back in, which would make
// the fused routine wait on itself. The block's terminator, such as the graph's
// output, doesn't count: it takes its operands only when the whole block ends.
// Siblings still join one blob, whether or not they share an input.
void formInBlock(Block &block, SymbolTable &symbols,
                 const llvm::DenseSet<Value> &invariant) {
  SmallVector<RoutineOp> blob;
  DeviceRefAttr blobDevice;
  // The values inside the blob: its members' results.
  llvm::DenseSet<Value> inBlob;
  // Ends the open blob, fusing it if it holds more than one routine.
  auto close = [&] {
    if (blob.size() > 1)
      fuse(blob);
    blob.clear();
    inBlob.clear();
  };

  for (Operation &op : llvm::make_early_inc_range(block)) {
    auto routine = dyn_cast<RoutineOp>(op);
    if (routine && canJoinBlob(routine, symbols, invariant)) {
      auto device = MO::getDeviceOrDefaultAny(routine);
      if (!blob.empty() && device != blobDevice)
        close();
      blobDevice = device;
      blob.push_back(routine);
      inBlob.insert(op.result_begin(), op.result_end());
      if (isReadOutside(routine, symbols, invariant))
        close();
      continue;
    }
    // A blob forms within one block and never spans blocks, so control flow
    // and `mo.parallel` keep working with blobs: the open blob ends here, and
    // each of their blocks forms its own.
    if (isa<RegionBranchOpInterface, MO::ParallelOp>(op)) {
      close();
      for (Region &region : op.getRegions()) {
        for (Block &nested : region) {
          if (sortTopologically(&nested))
            formInBlock(nested, symbols, invariant);
        }
      }
      continue;
    }
    // Holds by construction: `isReadOutside` ended the blob at any member with
    // a reader that can't join it, including one nested in a region.
    assert(
        (op.hasTrait<OpTrait::IsTerminator>() ||
         llvm::none_of(readValues(op),
                       [&](Value value) { return inBlob.contains(value); })) &&
        "an op outside the open blob reads its values");
  }
  close();
}

struct FormGraphBlobsPass
    : public impl::FormGraphBlobsBase<FormGraphBlobsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The kernel decls sit at the top of the module, beside the graphs, so the
    // pass runs on the module rather than on each graph.
    ModuleOp module = getOperation();
    SymbolTable &symbols =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    for (auto graph : module.getOps<MO::GraphOp>()) {
      // A device graph captures each launch as its own node.
      if (graph.getIsDeviceGraph())
        continue;
      // A graph region can use a value before the op that defines it. Both
      // the analysis and formation walk a block in dataflow order, so sort it
      // first; a block with a cycle has none and forms no blobs.
      Block &body = graph.getBody().front();
      if (!sortTopologically(&body))
        continue;
      llvm::DenseSet<Value> invariant = findInvariantValues(body);
      formInBlock(body, symbols, invariant);
    }
  }
};

} // namespace

} // namespace M::MOGG
