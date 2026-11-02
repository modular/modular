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
// Fuses sequences of declarative `mogg.routine`s into one routine whose body is
// a `mogg.graph_blob`, so that each sequence launches as one device kernel. The
// fused routine has every member's operands and results, and `OutlineRoutines`
// outlines it like any other.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_FORMGRAPHBLOBS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Whether `routine` can join a blob: its one call is to a declarative kernel,
// on tensors, named dims and one device context, with no chain and no results
// whose shape the call computes. `symbols` holds the kernel decls.
bool canJoinBlob(RoutineOp routine, SymbolTable &symbols) {
  // No chain, and no results whose shape the call computes.
  if (routine.getChain() || !routine.getOutputParamDecls().empty())
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

// Whether `next` continues the sequence that ends at `last`: it reads a result
// of `last`, nothing else reads `last`'s results, and both run on one device.
// The blob's intermediates then have no reader outside it, so fusing the
// sequence can't form a cycle with the ops around it.
//
// TODO(ehein): Let a blob hold a graph of routines, with branches and fan-out,
// not just a linear sequence.
bool extendsSequence(RoutineOp last, RoutineOp next) {
  if (MO::getDeviceOrDefaultAny(last) != MO::getDeviceOrDefaultAny(next))
    return false;
  bool readsLast = false;
  for (Value result : last.getOutputs()) {
    for (Operation *user : result.getUsers()) {
      if (user != next)
        return false;
      readsLast = true;
    }
  }
  return readsLast;
}

// Replaces `members` with one routine whose body is a `mogg.graph_blob` holding
// their bodies in order. It goes where the last member was, after every input
// of the sequence.
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

void formInBlock(Block &block, SymbolTable &symbols) {
  SmallVector<RoutineOp> sequence;
  // Ends the current sequence, fusing it if it holds more than one routine.
  auto close = [&] {
    if (sequence.size() > 1)
      fuse(sequence);
    sequence.clear();
  };
  for (Operation &op : llvm::make_early_inc_range(block)) {
    auto routine = dyn_cast<RoutineOp>(op);
    if (!routine) {
      for (Region &region : op.getRegions()) {
        for (Block &nested : region)
          formInBlock(nested, symbols);
      }
      continue;
    }
    if (!canJoinBlob(routine, symbols))
      continue;
    if (!sequence.empty() && extendsSequence(sequence.back(), routine)) {
      sequence.push_back(routine);
      continue;
    }
    close();
    sequence.push_back(routine);
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
    SymbolTable symbols(module);
    for (auto graph : module.getOps<MO::GraphOp>()) {
      // A device graph captures each launch as its own node.
      if (!graph.getIsDeviceGraph())
        formInBlock(graph.getBody().front(), symbols);
    }
  }
};

} // namespace

} // namespace M::MOGG
