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
#include "Support/AssertStream.h"
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

// Return a routine's outputs other than its trailing chain, if it has one.
ResultRange nonChainOutputs(RoutineOp routine) {
  ResultRange outputs = routine.getOutputs();
  return routine.getOutputChain() ? outputs.drop_back() : outputs;
}

// Whether `routine` can join a blob. `symbols` holds the kernel decls. It
// needs all of:
// - one call, to a declarative kernel on one device context;
// - operands that are tensors, buffers it writes in place under its chain, and
//   named dims;
// - no result with a data-dependent shape;
// - some output not in `invariant`. A routine computing only invariant values
//   runs once at init, and a blob runs on every execution.
bool canJoinBlob(RoutineOp routine, SymbolTable &symbols,
                 const llvm::DenseSet<Value> &invariant) {
  // No result with a data-dependent shape.
  if (!routine.getOutputParamDecls().empty())
    return false;

  // Some output varies, so a routine with no outputs stays out too.
  if (llvm::all_of(routine.getOutputs(),
                   [&](Value value) { return invariant.contains(value); }))
    return false;

  // Inputs are tensors or buffers, and outputs other than the chain are
  // tensors.
  auto isTensor = [](Type type) { return isa<MO::TensorType>(type); };
  [[maybe_unused]] auto isBuffer = [](Type type) {
    return isa<MO::BufferType>(type);
  };
  auto isTensorOrBuffer = [](Type type) {
    return isa<MO::TensorType, MO::BufferType>(type);
  };
  if (!llvm::all_of(routine.getInputs().getTypes(), isTensorOrBuffer) ||
      !llvm::all_of(nonChainOutputs(routine).getTypes(), isTensor))
    return false;
  ASSERT_STREAM(routine.getChain() ||
                    llvm::none_of(routine.getInputs().getTypes(), isBuffer),
                << "a routine that reads a buffer has a chain");

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

// Returns the chain that the routine fusing `members` waits on, or null if they
// wait on nothing. A member's chain joins values from earlier members and from
// outside the blob, directly or through the joins the blob absorbs,
// `chainJoins`. The blob runs its members in order, so it waits only on the
// values from outside, joined into one chain. A `mo.chain.create` takes any
// value, so an outside value can be a tensor, which a join makes a chain.
Value createBlobChain(OpBuilder &builder, Location loc,
                      ArrayRef<RoutineOp> members,
                      const llvm::SetVector<Operation *> &chainJoins) {
  llvm::DenseSet<Operation *> memberOps;
  for (RoutineOp member : members)
    memberOps.insert(member);
  llvm::SetVector<Value> waits;
  SmallVector<Value> worklist;
  for (RoutineOp member : llvm::reverse(members)) {
    if (Value chain = member.getChain())
      worklist.push_back(chain);
  }
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *def = value.getDefiningOp();
    if (def && memberOps.contains(def))
      continue;
    if (def && chainJoins.contains(def)) {
      llvm::append_range(worklist, llvm::reverse(def->getOperands()));
      continue;
    }
    waits.insert(value);
  }
  if (waits.empty())
    return {};
  if (waits.size() == 1 && isa<MO::ChainType>(waits.front().getType()))
    return waits.front();
  return MO::ChainCreateOp::create(builder, loc, waits.getArrayRef());
}

// Replaces `members` with one routine whose body is a `mogg.graph_blob` holding
// their bodies in order. It goes where the last member was, after every input
// of the blob. An absorbed chain join, from `chainJoins`, that something
// outside the blob still reads now reads the blob's chain, and may come before
// the routine: a graph region allows that, and `MOGGToMGP` sorts each block
// before lowering it. A join nothing else reads goes away.
void fuse(ArrayRef<RoutineOp> members,
          const llvm::SetVector<Operation *> &chainJoins) {
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
  llvm::DenseSet<Value> chains;
  bool hasReadEffect = false, hasWriteEffect = false;
  for (RoutineOp member : members) {
    for (Value input : member.getInputs()) {
      if (!resultIndex.contains(input) &&
          inputIndex.try_emplace(input, inputs.size()).second)
        inputs.push_back(input);
    }
    for (Value result : nonChainOutputs(member)) {
      resultIndex[result] = results.size();
      results.push_back(result);
    }
    if (Value chain = member.getOutputChain())
      chains.insert(chain);
    hasReadEffect |= member.getHasReadEffect();
    hasWriteEffect |= member.getHasWriteEffect();
  }

  Value chain = createBlobChain(rewriter, loc, members, chainJoins);

  SmallVector<Type> resultTypes(ValueRange(results).getTypes());
  if (!chains.empty())
    resultTypes.push_back(MO::ChainType::get(ctx));

  SmallVector<Operation *> memberOps(members.begin(), members.end());
  auto fused =
      RoutineOp::create(rewriter, loc, resultTypes, inputs, chain,
                        createFusionInfo<FusionSequenceAttr>(ctx, memberOps));
  fused.setHasReadEffect(hasReadEffect);
  fused.setHasWriteEffect(hasWriteEffect);
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
           llvm::zip_equal(nonChainOutputs(member), args.outputs))
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

  rewriter.replaceAllUsesWith(results, nonChainOutputs(fused));
  // Consumers of a chain produced within the blob now wait for the blob's
  // chain.
  for (Value memberChain : chains)
    rewriter.replaceAllUsesWith(memberChain, fused.getOutputChain());
  for (RoutineOp member : llvm::reverse(members))
    rewriter.eraseOp(member);
  for (Operation *join : llvm::reverse(chainJoins)) {
    if (join->use_empty())
      rewriter.eraseOp(join);
  }
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
// terminator, reads one of its results. A chain join stays inside the blob, so
// what reads the join counts instead.
bool isReadOutside(RoutineOp member, SymbolTable &symbols,
                   const llvm::DenseSet<Value> &invariant) {
  Block *block = member->getBlock();
  auto device = MO::getDeviceOrDefaultAny(member);
  SmallVector<Value> values(member->result_begin(), member->result_end());
  while (!values.empty()) {
    for (Operation *user : values.pop_back_val().getUsers()) {
      // A reader nested in a region, such as a `mo.if` branch, is outside:
      // only routines directly in `block` join its blobs.
      if (user->getBlock() != block)
        return true;
      if (user->hasTrait<OpTrait::IsTerminator>())
        continue;
      if (isa<MO::ChainCreateOp>(user)) {
        values.push_back(user->getResult(0));
        continue;
      }
      auto routine = dyn_cast<RoutineOp>(user);
      if (!routine || !canJoinBlob(routine, symbols, invariant) ||
          MO::getDeviceOrDefaultAny(routine) != device)
        return true;
    }
  }
  return false;
}

// The blob `formInBlock` keeps open, which it fuses when it closes.
struct OpenBlob {
  SmallVector<RoutineOp> members;
  DeviceRefAttr device;
  // The values inside the blob: its members' results and absorbed chain joins.
  llvm::DenseSet<Value> values;
  llvm::SetVector<Operation *> chainJoins;

  bool contains(Value value) const { return values.contains(value); }

  void add(RoutineOp member) {
    device = MO::getDeviceOrDefaultAny(member);
    members.push_back(member);
    values.insert(member->result_begin(), member->result_end());
  }

  void absorb(MO::ChainCreateOp join) {
    values.insert(join.getResult());
    chainJoins.insert(join);
  }

  // Ends the blob, fusing it if it holds more than one routine.
  void close() {
    if (members.size() > 1)
      fuse(members, chainJoins);
    members.clear();
    values.clear();
    chainJoins.clear();
  }
};

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
  OpenBlob blob;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    auto routine = dyn_cast<RoutineOp>(op);
    if (routine && canJoinBlob(routine, symbols, invariant)) {
      if (!blob.members.empty() &&
          MO::getDeviceOrDefaultAny(routine) != blob.device)
        blob.close();
      blob.add(routine);
      if (isReadOutside(routine, symbols, invariant))
        blob.close();
      continue;
    }
    // A join of the blob's chains orders a later member after earlier ones,
    // which the blob's own order already does, so it stays inside the blob.
    auto join = dyn_cast<MO::ChainCreateOp>(op);
    if (join && llvm::any_of(join->getOperands(), [&](Value value) {
          return blob.contains(value);
        })) {
      blob.absorb(join);
      continue;
    }
    // A blob forms within one block and never spans blocks, so control flow
    // and `mo.parallel` keep working with blobs: the open blob ends here, and
    // each of their blocks forms its own.
    if (isa<RegionBranchOpInterface, MO::ParallelOp>(op)) {
      blob.close();
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
    assert((op.hasTrait<OpTrait::IsTerminator>() ||
            llvm::none_of(readValues(op),
                          [&](Value value) { return blob.contains(value); })) &&
           "an op outside the open blob reads its values");
  }
  blob.close();
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
