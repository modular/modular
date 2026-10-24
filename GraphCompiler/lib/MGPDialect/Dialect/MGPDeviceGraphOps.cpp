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
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

/// The device a tensor/buffer graph input lives on. Both types default their
/// `#M.device_ref` parameter to the host, so an omitted device reads as CPU.
static DeviceRefAttr graphInputDevice(Value v) {
  return llvm::TypeSwitch<Type, DeviceRefAttr>(v.getType())
      .Case<MGP::DeviceTensorType, MGP::DeviceBufferType>(
          [](auto type) { return type.getDevice(); })
      .Default(nullptr);
}

/// Is `v` a device-resident graph input rather than a context operand?
/// (`inputs` also carries `!mgp.context` and `!mgp.device_context` values.)
///
/// Must agree with `graphInputData` in MGPPrimitives.cpp, or inputs pair with
/// the wrong stable buffers at replay. Mutable inputs register a null in-place
/// marker so every device input keeps one position in `graph->getInputs()`.
static bool isDeviceGraphInput(Value v) {
  return graphInputDevice(v) != nullptr;
}

/// Host tensor inputs are assumed to encode launch parameters, so their
/// contents join the cache key. Warn on large ones, which bloat the cache,
/// until there is a knob to control this.
static constexpr uint64_t kHostInputWarnBytes = 1024;

/// Verifies the residency rules for the graph's tensor/buffer inputs.
///
/// - Device-resident tensors are copied into device-pool stable locations (or
///   recorded in place when `mut`).
///
/// - Host-placed *immutable* tensors are allowed: they get a host-pool stable
///   location populated before recording, and their contents join the cache
///   key, because recorded ops read host values at enqueue time.
///
/// - Host-placed `mut` tensors are rejected (in-place writes would invalidate
///   their own cache entry), as are host-placed raw buffers, which should
///   already be lowered to mut tensors.
static LogicalResult verifyInputResidency(Operation *op, OperandRange inputs) {
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    DeviceRefAttr device = graphInputDevice(input);
    if (!device || !M::isHostPlaced(device))
      continue;

    auto tensorType = dyn_cast<MGP::DeviceTensorType>(input.getType());
    if (!tensorType)
      return op->emitOpError("input #")
             << idx << " is a buffer on " << device.toString()
             << ", but host-placed device graph inputs must be immutable "
                "tensors";
    if (tensorType.getMut())
      return op->emitOpError("input #")
             << idx << " is a mutable tensor on " << device.toString()
             << "; in-place writes contradict the value-keyed replay of a "
                "host-placed input";

    std::optional<uint64_t> byteSize =
        tensorType.getTensorSpec().getSizeInBytes();
    if (!byteSize)
      continue; // Dynamic shape: the runtime keys whatever size shows up.
    if (*byteSize > kHostInputWarnBytes)
      op->emitWarning("input #")
          << idx << " is a host tensor of " << *byteSize
          << " bytes; its contents are baked into the graph cache key, so "
             "every distinct value records a new graph";
  }
  return success();
}

// Emits `var cache_N = Pointer(to=<ctx>.get_device_graph_cache())`.
// `DeviceGraphCache` is not copyable, so the returned reference cannot be bound
// to a `var` directly; `DeviceGraph.create` takes it by pointer.
LogicalResult DeviceGraphCacheOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  MojoValue theContext = state.getMojoValueFor(getTheContext());
  MojoValue cache = state.builder.var(MojoValue(kStructPointer)(
      /*parameters=*/{},
      /*operands=*/{
          MojoValue(kParameterTo)
              .equal(StateContextRef(theContext).getDeviceGraphCache())}));
  state.setMojoValueFor(getOutCache(), cache);
  return success();
}

bool DeviceGraphCacheOp::isThrows() { return true; }

void DeviceGraphYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

// Packs each yielded value into an `AsyncValue` and passes it to `add_output`
// so its backing memory outlives the graph that references it.
LogicalResult DeviceGraphYieldOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  // The builder is the first block argument of the enclosing `create` body.
  Value builderArg = getOperation()->getBlock()->getArgument(0);
  DeviceGraphBuilderRef builder(state.getMojoValueFor(builderArg));

  for (Value output : getOutputs()) {
    MojoValue outVal = state.getMojoValueFor(output);
    MojoValue asyncValue =
        outVal.get(kFnAsyncPack)(/*parameters=*/{}, /*operands=*/{});
    state.builder.call(builder.addOutput(),
                       /*parameters=*/{}, /*operands=*/{asyncValue});
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphAllocOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphAllocOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  // The primitive's `sym_name` (set by ImportPackages) is the Mojo function.
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());

  DeviceRefAttr hostDevice = M::getHostDevice(getContext());
  MojoValue isHostParam(getOutBuffer().getType().getDevice() == hostDevice);
  MojoValue result = state.builder.var(function(
      /*parameters=*/{isHostParam},
      /*operands=*/{state.getMojoValueFor(getRuntimeSize()),
                    state.getMojoValueFor(getGraphBuilder())}));
  state.setMojoValueFor(getOutBuffer(), result);
  return success();
}

// The `mgp_device_graph_alloc` primitive raises (it allocates a buffer).
bool DeviceGraphAllocOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphRecordingContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.recording_context()`, which execute-only kernels
// dispatch their `execute` against.
LogicalResult
DeviceGraphRecordingContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.builder.var(builder.recordingContext());
  state.setMojoValueFor(getContext(), result);
  return success();
}

// `recording_context()` raises (it asks the driver for a recording context).
bool DeviceGraphRecordingContextOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphDeviceContextOp
//===----------------------------------------------------------------------===//

// Emits `var ctx_N = builder.context()`, a non-raising field read.
LogicalResult DeviceGraphDeviceContextOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue result = state.builder.var(builder.context());
  state.setMojoValueFor(getDeviceContext(), result);
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

DeviceGraphDependenciesYieldOp DeviceGraphDependenciesOp::getYieldOp() {
  return cast<DeviceGraphDependenciesYieldOp>(
      getBody().front().getTerminator());
}

TypedValue<DeviceGraphBuilderType>
DeviceGraphDependenciesOp::getBuilderBlockArgument() {
  return cast<TypedValue<DeviceGraphBuilderType>>(
      getBody().front().getArgument(0));
}

LogicalResult DeviceGraphDependenciesOp::verify() {
  // The sole block argument is the `!mgp.device_graph.builder` that the body's
  // device-context uses are re-sourced from.
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("body has ")
           << body.getNumArguments()
           << " block argument(s), expected exactly 1 (the "
              "`!mgp.device_graph.builder`)";
  if (!llvm::isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();

  // The yield's operands must match this op's `bodyResults`.
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();
  if (!llvm::equal(yieldOp.getOperands().getTypes(),
                   getBodyResults().getTypes()))
    return emitOpError(
        "`mgp.device_graph.dependencies.yield` operand types must "
        "match the op's body result types");
  return success();
}

LogicalResult DeviceGraphDependenciesOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  llvm::SmallSetVector<Value, 8> deduped;
  deduped.insert_range(getDependencies());

  if (deduped.size() == getDependencies().size())
    return failure();

  getDependenciesMutable().assign(deduped.getArrayRef());
  return success();
}

void DeviceGraphDependenciesOp::build(
    OpBuilder &builder, OperationState &result, Value graphBuilder,
    ValueRange dependencies,
    llvm::function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(graphBuilder);
  result.addOperands(dependencies);

  // The synthesized node is always the first result.
  result.addTypes(DeviceGraphNodeType::get(builder.getContext()));

  // The callback populates the body and is expected to emit the
  // dependencies.yield. The body otherwise references enclosing SSA directly.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value builderArg = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  if (bodyBuilder)
    bodyBuilder(builder, result.location, builderArg);

  // Supply the operandless terminator (no body results) if the callback did not
  // emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The remaining results are exactly the dependencies.yield operands.
  auto yieldOp = cast<DeviceGraphDependenciesYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperands().getTypes());
}

bool DeviceGraphDependenciesOp::producesNewNodes() {
  return !getBuilderBlockArgument().use_empty();
}

// Emits `var node_N = <builder>.region(work_N, dependencies=[...])`, where
// `work_N` is a recording closure wrapping the body. The body captures
// enclosing values lexically; only the builder block arg is a closure
// parameter. `dependencies=` makes the edges predecessors of the first node the
// body records (`_implicit_deps`); `region` chains the rest in body order.
//
// `region` returns only the body's last node, so each in-body result is
// surfaced through a captured outer `Optional[T]` var (the type may not be
// default-constructible, e.g. a non-null buffer pointer):
//
//   var out = Optional[T](None)
//   def work(mut builder) raises {mut}:
//     ...
//     out = <yielded>
//   var node = builder.region(work, dependencies=[...])
//   var resolved = out.take()
//   // uses of the op's body result now refer to `resolved`
LogicalResult DeviceGraphDependenciesOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  Value builderArg = getBuilderBlockArgument();
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();

  // Partition the used body results into those produced inside the body (which
  // need a captured outer var) and external pass-throughs (bound directly).
  struct SurfacedResult {
    Value result;
    Value yielded;
    MojoValue outVar;
  };

  SmallVector<SurfacedResult> surfaced;
  for (auto [result, yielded] :
       llvm::zip_equal(getBodyResults(), yieldOp.getOperands())) {
    if (result.use_empty())
      continue;

    MojoValue optionalType = MojoValue(
        kStructOptional)[EmitMojoState::mlirTypeToMojoType(result.getType())];
    MojoValue outVar =
        state.builder.var(optionalType(/*parameters=*/{},
                                       /*operands=*/{MojoValue(kComptimeNone)}),
                          /*cache=*/false);
    surfaced.emplace_back(result, yielded, outVar);
  }

  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", MojoValue(kStructDeviceGraphBuilder));
  builderParam.argConvention = "mut";

  // Surfacing closures write their `Optional` vars, so capture by `mut`. One
  // that also reads register-passable captures would need per-capture
  // conventions; current single-dispatch bodies do not.
  llvm::StringRef workConvention =
      surfaced.empty() ? llvm::StringRef("imm") : llvm::StringRef("mut");

  bool errOccurred = false;
  MojoValue workClosure = state.builder.lambda(
      "work", /*parameters=*/{}, /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt,
      // Never inline: single-use per-kernel bodies otherwise merge into the
      // build closure (see the build lambda in DeviceGraphCreateOp).
      MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/workConvention,
      [&](const FunctionBody &functionBody) {
        state.setMojoValueFor(builderArg, functionBody.arguments.front());
        for (Operation &op : getBodyBlock().getOperations())
          if (state.lowerOpToMojo(&op).failed()) {
            errOccurred = true;
            return;
          }
        for (const SurfacedResult &s : surfaced)
          state.builder.assign(s.outVar,
                               state.getMojoValueFor(s.yielded).transfer());
      });
  if (errOccurred)
    return failure();

  SmallVector<MojoValue> depValues = state.getMojoValuesFor(getDependencies());
  MojoValue dependenciesList = state.builder.listLiteral(depValues);

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  MojoValue node =
      state.builder.var(builder.region(workClosure, dependenciesList));
  state.setMojoValueFor(getNode(), node);

  for (const SurfacedResult &s : surfaced)
    state.setMojoValueFor(s.result, state.builder.var(s.outVar.get("take")(
                                        /*parameters=*/{}, /*operands=*/{})));

  return success();
}

Block &DeviceGraphDependenciesOp::getBodyBlock() { return getBody().front(); }

bool DeviceGraphDependenciesOp::isThrows() { return true; }

namespace {

/// Rebuilds an `mgp.device_graph.dependencies` op with new body results and
/// yield operands, keeping the `node` result and dependency edges.
/// `bodyResultReplacement(i)` maps original body result `i` to its position
/// among the new body results.
DeviceGraphDependenciesOp
rebuildWithBodyResults(PatternRewriter &rewriter, DeviceGraphDependenciesOp op,
                       TypeRange newBodyResultTypes,
                       ValueRange newYieldOperands,
                       ArrayRef<unsigned> bodyResultRemap) {
  SmallVector<Type> newResultTypes;
  newResultTypes.push_back(op.getNode().getType());
  llvm::append_range(newResultTypes, newBodyResultTypes);

  auto newOp = DeviceGraphDependenciesOp::create(
      rewriter, op.getLoc(), TypeRange(newResultTypes), op.getGraphBuilder(),
      op.getDependencies());
  newOp.getBody().takeBody(op.getBody());
  DeviceGraphDependenciesYieldOp newYield = newOp.getYieldOp();
  rewriter.modifyOpInPlace(newYield, [&]() {
    newYield.getOperandsMutable().assign(newYieldOperands);
  });

  SmallVector<Value> replacements(op.getNumResults());
  replacements[0] = newOp.getNode();
  for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
    if (result.use_empty())
      continue;
    replacements[idx + 1] = newOp.getResult(bodyResultRemap[idx] + 1);
  }
  rewriter.replaceOp(op, replacements);
  return newOp;
}

/// Drops unused body results of an `mgp.device_graph.dependencies` op and the
/// matching yield operands. The `node` result (the ordering edge) is kept.
struct RemoveUnusedDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == op.getBodyResults().size())
      return failure();

    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    SmallVector<unsigned> remap(op.getBodyResults().size());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      remap[oldIdx] = newIdx;

    rebuildWithBodyResults(rewriter, op, TypeRange(usedResultTypes),
                           newYieldOperands, remap);
    return success();
  }
};

/// De-duplicates body results of an `mgp.device_graph.dependencies` op whose
/// yield carries the same value more than once.
struct DeduplicateDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();

    unsigned numElements = oldYield.getNumOperands();

    llvm::SmallMapVector<Value, unsigned, 8> firstOccurrence;
    firstOccurrence.reserve(numElements);

    SmallVector<unsigned> resultRemap(op.getBodyResults().size());

    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, _] =
          firstOccurrence.try_emplace(operand, firstOccurrence.size());
      resultRemap[idx] = it->second;
    }

    if (firstOccurrence.size() == op.getBodyResults().size())
      return failure();

    auto dedupedOperands = llvm::to_vector(firstOccurrence.keys());
    rebuildWithBodyResults(rewriter, op, TypeRange(dedupedOperands),
                           dedupedOperands, resultRemap);
    return success();
  }
};

/// Forwards `mgp.device_graph.dependencies` body results whose
/// `dependencies.yield` operand is produced *outside* the body region.
/// Rewriting those uses leaves the result dead for
/// `RemoveUnusedDependenciesResults` to drop.
struct ForwardExternalDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    Region &body = op.getBody();
    DeviceGraphDependenciesYieldOp yieldOp = op.getYieldOp();

    bool changed = false;
    for (auto [result, yielded] :
         llvm::zip(op.getBodyResults(), yieldOp.getOperands())) {
      // Skipping dead results keeps the pattern idempotent; otherwise the
      // greedy driver never converges.
      if (result.use_empty())
        continue;

      // Only forward pass-throughs of outer SSA (block args count as inside).
      if (body.isAncestor(yielded.getParentRegion()))
        continue;

      rewriter.replaceAllUsesWith(result, yielded);
      changed = true;
    }
    return success(changed);
  }
};

/// Hoists body ops whose operands are all defined outside the body in front of
/// an `mgp.device_graph.dependencies` op, so the `work` closure captures them
/// once instead of recomputing them on every graph build. Hoisting is
/// transitive within one sweep; builder-dependent ops stay in the body.
struct HoistOperations final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    Region &body = dependenciesOp.getBody();

    auto isRegionCapture = [&body](Value operand) {
      return body.isAncestor(operand.getParentRegion());
    };

    bool changed = false;
    for (Operation &op : llvm::make_early_inc_range(body.getOps())) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      WalkResult result = op.walk([&](Operation *inner) -> WalkResult {
        bool canHoist = llvm::none_of(inner->getOperands(), isRegionCapture);

        if (!canHoist)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

      if (!result.wasInterrupted()) {
        rewriter.moveOpBefore(&op, dependenciesOp);
        changed = true;
      }
    }

    return success(changed);
  }
};

/// Absorbs the body of an `mgp.device_graph.dependencies` op into its sole
/// successor, when the two form a link in a linear chain.
struct MergeChainedDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    // With exactly one edge, absorbing the producer adds no ordering. With
    // more, the absorbed body would be ordered after the other edges too.
    if (op.getDependencies().size() != 1)
      return failure();

    auto producer =
        op.getDependencies().front().getDefiningOp<DeviceGraphDependenciesOp>();
    if (!producer)
      return failure();

    if (!producer.getNode().hasOneUse())
      return failure();

    // Dataflow out of the producer is uncommon; bail.
    if (!producer.getBodyResults().empty())
      return failure();

    if (producer.getGraphBuilder() != op.getGraphBuilder() ||
        producer->getBlock() != op->getBlock())
      return failure();

    // The merged block keeps the consumer's terminator.
    rewriter.eraseOp(producer.getYieldOp());

    rewriter.inlineBlockBefore(&producer.getBodyBlock(), &op.getBodyBlock(),
                               op.getBodyBlock().begin(),
                               {op.getBuilderBlockArgument()});

    rewriter.modifyOpInPlace(op, [&] {
      op.getDependenciesMutable().assign(producer.getDependencies());
    });
    rewriter.eraseOp(producer);

    return success();
  }
};

/// Replaces a dependency edge on an empty (node-less) producer with that
/// producer's own edges, preserving transitive ordering. Bypassed ops are left
/// for DCE; duplicate edges are folded by the edge-dedup folder.
struct BypassEmptyDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  /// Returns the transitive dependencies of a node for an empty region.
  static std::optional<ValueRange> getTransitiveDependencies(Value dependency) {
    if (auto producer = dependency.getDefiningOp<DeviceGraphDependenciesOp>())
      if (!producer.producesNewNodes())
        return producer.getDependencies();

    return {};
  }

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newDependencies;

    bool changed = false;
    for (auto dependency : dependenciesOp.getDependencies()) {
      if (auto transitiveDeps = getTransitiveDependencies(dependency)) {
        llvm::append_range(newDependencies, *transitiveDeps);
        changed = true;
      } else {
        newDependencies.push_back(dependency);
      }
    }

    if (!changed)
      return rewriter.notifyMatchFailure(dependenciesOp,
                                         "no dependencies to remove");

    rewriter.modifyOpInPlace(dependenciesOp, [&] {
      dependenciesOp.getDependenciesMutable().assign(newDependencies);
    });

    return success();
  }
};

} // namespace

void DeviceGraphDependenciesOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.add<
    BypassEmptyDependencies,
    DeduplicateDependenciesResults,
    ForwardExternalDependenciesResults,
    HoistOperations,
    MergeChainedDependencies,
    RemoveUnusedDependenciesResults
  >(ctx);
  // clang-format on
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphDependenciesYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

// The parent op's `emitMojo` surfaces the body results.
LogicalResult DeviceGraphDependenciesYieldOp::emitMojo(MOGG::EmitMojoState &) {
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

DeviceGraphStagingYieldOp DeviceGraphStagingOp::getYieldOp() {
  return cast<DeviceGraphStagingYieldOp>(getBody().front().getTerminator());
}

LogicalResult DeviceGraphStagingOp::verify() {
  Block &body = getBody().front();

  // Block args are [builder, input_0, ..., input_n-1]; inputs carry the
  // `inputs` operand types verbatim (bare `T`, no `!mogg.async`).
  if (body.getNumArguments() != 1 + getInputs().size())
    return emitOpError("staging body has ")
           << body.getNumArguments() << " block argument(s), expected "
           << (1 + getInputs().size())
           << " (the device-graph builder plus one "
              "per input)";
  if (!isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("staging body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();
  for (auto [idx, inputType] : llvm::enumerate(getInputs().getTypes())) {
    Type argType = body.getArgument(idx + 1).getType();
    if (argType != inputType)
      return emitOpError("staging body argument #")
             << (idx + 1) << " has type " << argType << ", expected "
             << inputType;
  }

  DeviceGraphStagingYieldOp yieldOp = getYieldOp();
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("`mgp.device_graph.staging.yield` has ")
           << yieldOp.getNumOperands() << " operand(s), expected "
           << getNumResults();
  for (auto [idx, resultType] :
       llvm::enumerate(getStagedResults().getTypes())) {
    Type operandType = yieldOp.getOperand(idx).getType();
    if (operandType != resultType)
      return emitOpError("`mgp.device_graph.staging.yield` operand #")
             << idx << " has type " << operandType << ", expected "
             << resultType;
  }
  return success();
}

void DeviceGraphStagingOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, Value, ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(inputs);

  // The callback populates the body and is expected to emit the staging.yield.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  SmallVector<Value> inputArgs;
  inputArgs.reserve(inputs.size());
  for (Value input : inputs)
    inputArgs.push_back(body->addArgument(input.getType(), result.location));
  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, inputArgs);

  // Supply the operandless terminator (zero staged results) if the callback did
  // not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  auto yieldOp = cast<DeviceGraphStagingYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperandTypes());
}

namespace {

/// Drops unused results of an `mgp.device_graph.staging` op and the matching
/// yield operands.
struct RemoveUnusedStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == op.getNumResults())
      return failure();

    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(usedResultTypes), op.getInputs(),
        op.getConstantResources());

    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(newYieldOperands);
    });

    SmallVector<Value> replacements(op.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      replacements[oldIdx] = newOp.getResult(newIdx);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// De-duplicates results of an `mgp.device_graph.staging` op whose yield
/// carries the same value more than once.
struct DeduplicateStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();

    llvm::SmallDenseMap<Value, unsigned> firstOccurrence;
    SmallVector<Value> dedupedOperands;
    SmallVector<Type> dedupedTypes;
    SmallVector<unsigned> resultRemap(op.getNumResults());
    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, inserted] =
          firstOccurrence.try_emplace(operand, dedupedOperands.size());
      if (inserted) {
        dedupedOperands.push_back(operand);
        dedupedTypes.push_back(op.getResult(idx).getType());
      }
      resultRemap[idx] = it->second;
    }

    if (dedupedOperands.size() == op.getNumResults())
      return failure();

    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(dedupedTypes), op.getInputs(),
        op.getConstantResources());

    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(dedupedOperands);
    });

    SmallVector<Value> replacements =
        llvm::map_to_vector(resultRemap, [&](unsigned newIdx) -> Value {
          return newOp.getResult(newIdx);
        });
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void DeviceGraphStagingOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RemoveUnusedStagingResults, DeduplicateStagingResults>(ctx);
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphStagingYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStableCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = cast<DeviceTensorType>(adaptor.getTensor().getType());

  auto resultType = DeviceTensorType::get(
      tensorType.getDevice(), tensorType.getTensorSpec(), tensorType.getMut(),
      /*stable=*/true);

  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult DeviceGraphStableCreateOp::verify() {
  if (getTensor().getType().getStable())
    return emitOpError("input tensor is already marked `stable`");

  size_t numDynamicDims =
      getTensor().getType().getTensorSpec().getNumDynamicDims();
  if (getRuntimeDims().size() != numDynamicDims)
    return emitOpError("expected ")
           << numDynamicDims << " runtime dimension(s), one per dynamic "
           << "dimension of the tensor, but got " << getRuntimeDims().size();
  return success();
}

bool DeviceGraphStableCreateOp::mayAlias(size_t operandIndex,
                                         size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableGetOp
//===----------------------------------------------------------------------===//

/// The `stableTypes` of the nearest collective enclosing `op`, or null when
/// there is none (dialect-level tests write the ops standalone). Lowering
/// wraps the op in compiled regions and dependency units on its way to
/// emission, so this walks rather than checking the immediate parent.
///
/// TODO(spenser): carry the slot types on `!mgp.device_graph.builder` itself,
/// so `stable.get`/`stable.place` verify against their operand's type and the
/// walk (and the collective-side annotation) go away.
static StableAddressArrayAttr enclosingStableTypes(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto collective = dyn_cast<DeviceGraphCollectiveOp>(parent))
      return collective.getStableTypesAttr();
    if (auto create = dyn_cast<DeviceGraphCollectiveCreateOp>(parent))
      return create.getStableTypesAttr();
  }
  return {};
}

/// Checks that slot `index` exists in the enclosing collective and is declared
/// with `slotType`. Slots are named by index rather than by SSA value, so this
/// is the only place the two agree.
static LogicalResult verifyStableSlot(Operation *op, int64_t index,
                                      Type slotType) {
  if (index < 0)
    return op->emitOpError("stable slot index must not be negative");

  StableAddressArrayAttr stableTypes = enclosingStableTypes(op);
  if (!stableTypes)
    return success();

  ArrayRef<StableAddressType> slots = stableTypes.getValue();
  if (static_cast<uint64_t>(index) >= slots.size())
    return op->emitOpError("stable slot index ")
           << index << " is out of range; the enclosing collective declares "
           << slots.size() << " slot(s)";
  if (slots[index] != slotType)
    return op->emitOpError("stable slot #")
           << index << " is declared as " << slots[index]
           << ", but the op uses " << slotType;
  return success();
}

bool DeviceGraphStableGetOp::isThrows() { return true; }

LogicalResult DeviceGraphStableGetOp::verify() {
  return verifyStableSlot(*this, getIndex(), getResult().getType());
}

LogicalResult DeviceGraphStableGetOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  state.setMojoValueFor(
      getResult(), state.builder.var(builder.getStable(MojoValue(getIndex()))));
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStablePlaceOp
//===----------------------------------------------------------------------===//

bool DeviceGraphStablePlaceOp::isThrows() { return true; }

LogicalResult DeviceGraphStablePlaceOp::verify() {
  return verifyStableSlot(*this, getIndex(), getPlacedAddr().getType());
}

LogicalResult DeviceGraphStablePlaceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  // The slot holds the tensor's data pointer. An `!mgp.tensor` value has no
  // pointer accessor of its own, so take it from the extracted buffer, as
  // `mogg.tensor.__init__` does.
  auto extractDecl =
      state.registry.lookupPrimitive(TensorExtractBufferOp::getOperationName());
  MojoValue extractBuffer(extractDecl.getSymNameAttr().strref());
  MojoValue buffer = state.builder.var(extractBuffer(
      /*parameters=*/{}, {state.getMojoValueFor(getTensor())}));
  MojoValue dataPtr =
      OpaquePointerRef(ManagedTensorSliceRef(buffer).unsafePtr())
          .bitcast(MojoValue(kComptimeDTypeNoneType))();

  DeviceGraphBuilderRef builder(state.getMojoValueFor(getGraphBuilder()));
  state.setMojoValueFor(getPlacedAddr(), state.builder.var(builder.placeStable(
                                             MojoValue(getIndex()), dataPtr)));
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveOp
//===----------------------------------------------------------------------===//

// Verifies:
//   1. `$devices.size() == #regions`.
//   2. Every region has exactly one block argument, a
//      `!mgp.device_graph.builder`.
//   3. Every region terminates with `DeviceGraphCollectiveYieldOp`.
//   4. The total yielded operand count across regions equals `$outputs.size()`
//      (yields append in region order).
// `$stableTypes` needs no check here: its attribute type only admits
// `!mgp.stable_addr` elements.
LogicalResult DeviceGraphCollectiveOp::verify() {
  if (getGraphs().size() != getDevices().size())
    return emitOpError("`devices` attribute has ")
           << getDevices().size() << " entries, but the op has "
           << getGraphs().size() << " regions";

  if (getDevices().empty())
    return success();

  unsigned outputIdx = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.getNumArguments() != 1)
      return emitOpError("region #")
             << idx << " has " << block.getNumArguments()
             << " block argument(s); expected exactly one, the "
                "`!mgp.device_graph.builder`";

    if (!isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << idx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    if (!block.mightHaveTerminator() ||
        !isa<DeviceGraphCollectiveYieldOp>(block.getTerminator()))
      return emitOpError("region #")
             << idx
             << " must terminate with `mgp.device_graph.collective.yield`";

    auto yield = cast<DeviceGraphCollectiveYieldOp>(block.getTerminator());
    for (auto operand : yield.getOperands()) {
      if (outputIdx >= getOutputs().size())
        return emitOpError("region #")
               << idx << " yields more operands than declared outputs ("
               << getOutputs().size() << ")";
      if (operand.getType() != getOutputs()[outputIdx].getType())
        return emitOpError("region #")
               << idx << " yield operand type " << operand.getType()
               << " does not match output #" << outputIdx << " type "
               << getOutputs()[outputIdx].getType();
      ++outputIdx;
    }
  }

  if (outputIdx != getOutputs().size())
    return emitOpError("total `mgp.device_graph.collective.yield` operand "
                       "count across regions is ")
           << outputIdx << ", expected " << getOutputs().size()
           << " (one per op result)";

  return success();
}

ParseResult DeviceGraphCollectiveOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  // The shared builder and stable-address block arguments are threaded into
  // every per-device region.
  if (parser.parseKeyword("builder"))
    return failure();

  if (parser.parseLParen())
    return failure();

  OpAsmParser::Argument builderArg;
  if (parser.parseArgument(builderArg, /*allowType=*/false))
    return failure();

  // The builder type is optional and defaults to `!mgp.device_graph.builder`.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(builderArg.type))
      return failure();
  } else {
    builderArg.type = DeviceGraphBuilderType::get(parser.getContext());
  }

  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("stable"))
    return failure();

  SmallVector<StableAddressType> stableTypes;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            return parser.parseType(stableTypes.emplace_back());
          }))
    return failure();

  // Every region's entry block takes just the builder.
  SmallVector<OpAsmParser::Argument> regionArgs = {builderArg};

  // Each region is a `DeviceRefAttr` label then its body; label order defines
  // `$devices`.
  SmallVector<DeviceRefAttr> deviceRefs;

  auto parseDeviceRegionPair = [&]() -> ParseResult {
    DeviceRefAttr deviceRef;
    if (parser.parseCustomAttributeWithFallback(deviceRef))
      return failure();
    deviceRefs.push_back(deviceRef);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/false))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                     parseDeviceRegionPair))
    return failure();

  // `devices` comes from the region labels, so drop it from the attr-dict.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.attributes.erase(getDevicesAttrName(result.name));
  result.attributes.erase(getStableTypesAttrName(result.name));

  auto devices = DeviceRefsAttr::get(parser.getContext(), deviceRefs);
  result.addAttribute(getDevicesAttrName(result.name), devices);
  result.addAttribute(
      getStableTypesAttrName(result.name),
      StableAddressArrayAttr::get(parser.getContext(), stableTypes));

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveOp::print(OpAsmPrinter &p) {
  // Print the shared entry-block arguments once, from the first region; the
  // region bodies are then printed without them.
  p << " builder(";
  if (!getGraphs().empty()) {
    p.printRegionArgument(getRegionBlock(0).getArgument(0),
                          /*argAttrs=*/{}, /*omitType=*/true);
  }
  p << ") stable(";
  llvm::interleaveComma(getStableTypes(), p,
                        [&](StableAddressType type) { p.printType(type); });
  p << ')';

  if (!getDevices().empty())
    p << " ";

  llvm::interleaveComma(llvm::enumerate(getDevices()), p, [&](auto idxAndAttr) {
    auto [idx, attr] = idxAndAttr;
    p.printStrippedAttrOrType(attr);
    p << " ";
    p.printRegion(getGraphs()[idx], /*printEntryBlockArgs=*/false);
  });

  // `devices` and `stableTypes` are represented by the per-region labels and
  // the `stable(...)` clause above.
  SmallVector<StringRef> elide = {getDevicesAttrName(),
                                  getStableTypesAttrName()};
  if (getConstantResources().empty())
    elide.push_back(getConstantResourcesAttrName());
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  if (!getOutputs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutputs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // The only entry block argument is the builder, `b`.
  for (BlockArgument arg : region.getArguments())
    setNameFn(arg, "b");
}

void DeviceGraphCollectiveOp::build(
    OpBuilder &builder, OperationState &state,
    ArrayRef<M::DeviceRefAttr> devices, ArrayRef<StableAddressType> stableTypes,
    function_ref<void(OpBuilder &, M::DeviceRefAttr, BlockArgument)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  MLIRContext *ctx = builder.getContext();

  // Build one region per device, each with the builder as its only block
  // argument.
  SmallVector<Type> resultTypes;
  for (auto dev : devices) {
    Region *region = state.addRegion();
    Block *block = builder.createBlock(region);

    auto builderArg =
        block->addArgument(DeviceGraphBuilderType::get(ctx), state.location);

    builder.setInsertionPointToEnd(block);
    bodyBuilder(builder, dev, builderArg);
    ensureTerminator(*region, builder, state.location);

    llvm::append_range(resultTypes, block->getTerminator()->getOperandTypes());
  }

  auto stableTypesAttr = StableAddressArrayAttr::get(ctx, stableTypes);

  // Call into the tablegen builder with `graphsCount=0`. We've already added
  // the child regions to `state`.
  DeviceGraphCollectiveOp::build(builder, state, resultTypes, devices,
                                 stableTypesAttr,
                                 /*constantResources=*/ArrayAttr(),
                                 /*graphsCount=*/0);
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableTensorOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphStableTensorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DeviceTensorType tensorType =
      cast<StableAddressType>(adaptor.getAddr().getType()).getElementType();
  inferredReturnTypes.push_back(DeviceTensorType::get(
      tensorType.getDevice(), tensorType.getTensorSpec(), tensorType.getMut(),
      /*stable=*/true));
  return success();
}

LogicalResult DeviceGraphStableTensorOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primDecl.getSymNameAttr().strref());

  // Parameterized like `mgp.tensor.create`: the result is an ordinary
  // `OwnedTensor` whose data pointer happens to be the slot.
  DeviceTensorType tensorType = getResult().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  MojoValue dtypeParam = state.mlirDTypeToMojoType(
      tensorType.getTensorSpec().getOptDType().value());

  SmallVector<MojoValue> params = {tensorRankParam, bufferRankParam,
                                   dtypeParam};
  // `kind` names how the tensor participates as a device graph input; the
  // Mojo default (`data`) is left implicit.
  if (tensorType.getMut())
    params.push_back(
        MojoValue(kParameterKind)
            .equal(
                MojoValue(kStructDeviceGraphInputKind).get(kInputKindMutable)));
  else if (M::isHostPlaced(tensorType.getDevice()))
    params.push_back(
        MojoValue(kParameterKind)
            .equal(MojoValue(kStructDeviceGraphInputKind).get(kInputKindHost)));

  state.setMojoValueFor(getResult(),
                        state.builder.var(function(
                            params, state.getMojoValuesFor(getOperands()))));
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphCollectiveYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveCreateOp
//===----------------------------------------------------------------------===//

ParseResult DeviceGraphCollectiveCreateOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  Type contextType = MGP::ContextType::get(ctx);

  // The shared `!mgp.context` is also each region's second block argument.
  OpAsmParser::UnresolvedOperand theContext;
  if (parser.parseOperand(theContext))
    return failure();

  // cache(%cache : !mgp.device_graph.cache)?
  bool hasCache = false;
  OpAsmParser::UnresolvedOperand cacheVal;
  Type cacheType;
  if (succeeded(parser.parseOptionalKeyword("cache"))) {
    hasCache = true;
    if (parser.parseLParen() || parser.parseOperand(cacheVal) ||
        parser.parseColon() || parser.parseType(cacheType) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> deviceContextVals;
  SmallVector<Type> deviceContextTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> inputVals;
  SmallVector<Type> inputTypes;

  // The body omits its entry-block arguments; the prefix clauses declare them.
  auto parseGraph = [&]() -> ParseResult {
    // graph(<deviceContext> : <type>)
    if (parser.parseKeyword("graph") || parser.parseLParen())
      return failure();

    OpAsmParser::UnresolvedOperand deviceContext;
    Type deviceContextType;
    if (parser.parseOperand(deviceContext) || parser.parseColon() ||
        parser.parseType(deviceContextType) || parser.parseRParen())
      return failure();
    deviceContextVals.push_back(deviceContext);
    deviceContextTypes.push_back(deviceContextType);

    // builder(<name>)
    if (parser.parseKeyword("builder") || parser.parseLParen())
      return failure();
    OpAsmParser::Argument builderArg;
    if (parser.parseArgument(builderArg, /*allowType=*/false))
      return failure();
    builderArg.type = DeviceGraphBuilderType::get(ctx);
    if (parser.parseRParen())
      return failure();

    // inputs(<name> : <type>, ...)
    if (parser.parseKeyword("inputs") || parser.parseLParen())
      return failure();
    SmallVector<OpAsmParser::Argument> inputArgs;
    if (succeeded(parser.parseOptionalRParen())) {
      // empty input list
    } else {
      do {
        OpAsmParser::UnresolvedOperand input;
        Type inputType;
        if (parser.parseOperand(input) || parser.parseColon() ||
            parser.parseType(inputType))
          return failure();
        inputVals.push_back(input);
        inputTypes.push_back(inputType);

        OpAsmParser::Argument inputArg;
        inputArg.ssaName = input;
        inputArg.type = inputType;
        inputArgs.push_back(inputArg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }

    // Reuse the op-level context operand's name for the block argument.
    OpAsmParser::Argument contextArg;
    contextArg.ssaName = theContext;
    contextArg.type = contextType;

    SmallVector<OpAsmParser::Argument> regionArgs;
    regionArgs.push_back(builderArg);
    regionArgs.push_back(contextArg);
    llvm::append_range(regionArgs, inputArgs);

    Region *region = result.addRegion();
    if (parser.parseRegion(*region, regionArgs,
                           /*enableNameShadowing=*/true))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(parseGraph))
    return failure();

  if (parser.resolveOperand(theContext, contextType, result.operands) ||
      parser.resolveOperands(deviceContextVals, deviceContextTypes, {},
                             result.operands) ||
      (hasCache &&
       parser.resolveOperand(cacheVal, cacheType, result.operands)) ||
      parser.resolveOperands(inputVals, inputTypes, {}, result.operands))
    return failure();

  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(deviceContextVals.size()), hasCache ? 1 : 0,
           static_cast<int32_t>(inputVals.size())}));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  return success();
}

void DeviceGraphCollectiveCreateOp::print(OpAsmPrinter &p) {
  p << ' ' << getTheContext();

  if (Value cache = getCache()) {
    p << " cache(" << cache << " : " << cache.getType() << ")";
  }

  size_t inputOffset = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    if (idx != 0)
      p << ",";

    Block &body = region.front();
    Value deviceContext = getDeviceContexts()[idx];
    BlockArgument builderArg = body.getArgument(0);
    size_t numRegionInputs = body.getNumArguments() - 2;
    auto regionInputs =
        getInputs().drop_front(inputOffset).take_front(numRegionInputs);

    p << " graph(" << deviceContext << " : " << deviceContext.getType()
      << ") builder(";
    p.printRegionArgument(builderArg, /*argAttrs=*/{}, /*omitType=*/true);

    p << ") inputs(";
    llvm::interleaveComma(regionInputs, p, [&](Value input) {
      p << input << " : " << input.getType();
    });
    p << ") ";

    SmallVector<Value> shadowNames;
    shadowNames.reserve(1 + 1 + numRegionInputs);
    shadowNames.push_back(builderArg);
    shadowNames.push_back(getTheContext());
    llvm::append_range(shadowNames, regionInputs);
    p.shadowRegionArgs(region, shadowNames);

    p.printRegion(region, /*printEntryBlockArgs=*/false);
    inputOffset += numRegionInputs;
  }

  SmallVector<StringRef> elide = {getOperandSegmentSizesAttrName()};
  p.printOptionalAttrDict(getOperation()->getAttrs(), elide);

  if (!getOutGraphs().empty()) {
    p << " : ";
    llvm::interleaveComma(getOutGraphs(), p,
                          [&](Value v) { p.printType(v.getType()); });
  }
}

void DeviceGraphCollectiveCreateOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  unsigned idx = region.getRegionNumber();
  setNameFn(region.front().getArgument(0), "b" + Twine(idx).str());
}

LogicalResult DeviceGraphCollectiveCreateOp::verify() {
  if (std::optional<ArrayRef<int64_t>> shapeOnly = getShapeOnlyInputs()) {
    for (int64_t index : *shapeOnly) {
      if (index < 0 || static_cast<size_t>(index) >= getInputs().size())
        return emitOpError("`shapeOnlyInputs` names input #")
               << index << ", but the op has " << getInputs().size()
               << " inputs";
      if (!graphInputDevice(getInputs()[index]))
        return emitOpError("`shapeOnlyInputs` names input #")
               << index << ", which is not a tensor or buffer";
    }
  }

  if (getGraphs().size() != getDeviceContexts().size())
    return emitOpError("`deviceContexts` has ")
           << getDeviceContexts().size() << " operands, but the op has "
           << getGraphs().size() << " regions";

  if (getOutGraphs().size() != getGraphs().size())
    return emitOpError() << "number of result graphs (" << getOutGraphs().size()
                         << ") does not match the number of body regions ("
                         << getGraphs().size() << ")";

  if (getGraphs().empty())
    return emitOpError("must have at least one device graph region");

  Type theContextType = getTheContext().getType();
  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  size_t inputIdx = 0;

  for (auto [regionIdx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.empty() || !llvm::isa<DeviceGraphYieldOp>(block.back()))
      return emitOpError("region #")
             << regionIdx << " must terminate with `mgp.device_graph.yield`";

    if (block.getNumArguments() < 2)
      return emitOpError("region #")
             << regionIdx << " has " << block.getNumArguments()
             << " block argument(s), expected at least 2 (the builder and the "
                "context)";

    if (!llvm::isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << regionIdx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    if (block.getArgument(1).getType() != theContextType)
      return emitOpError("region #")
             << regionIdx
             << " block argument #1 must be the `theContext` operand type "
             << theContextType << ", but is " << block.getArgument(1).getType();

    size_t numRegionInputs = block.getNumArguments() - 2;
    if (inputIdx + numRegionInputs > getInputs().size())
      return emitOpError("region #")
             << regionIdx << " has " << numRegionInputs
             << " input block argument(s), but only "
             << (getInputs().size() - inputIdx) << " input operand(s) remain";

    for (size_t i = 0; i < numRegionInputs; ++i) {
      Type argType = block.getArgument(2 + i).getType();
      Type inputType = getInputs()[inputIdx + i].getType();
      if (argType != inputType)
        return emitOpError("region #")
               << regionIdx << " block argument #" << (2 + i) << " has type "
               << argType << ", expected " << inputType;
    }

    auto yieldOp = cast<DeviceGraphYieldOp>(block.back());
    auto graphType = cast<DeviceGraphType>(getOutGraphs()[regionIdx].getType());
    FunctionType sig = graphType.getSignature();

    ArrayRef<Type> regionInputTypes(
        ArrayRef<Type>(inputTypes).slice(inputIdx, numRegionInputs));
    if (!llvm::equal(sig.getInputs(), regionInputTypes))
      return emitOpError("result graph #")
             << regionIdx
             << " signature inputs do not match this region's input operands";

    if (!llvm::equal(sig.getResults(), yieldOp.getOutputs().getTypes()))
      return emitOpError("result graph #")
             << regionIdx
             << " signature results do not match the region's yielded value "
                "types";

    inputIdx += numRegionInputs;
  }

  if (inputIdx != getInputs().size())
    return emitOpError("the op has ")
           << getInputs().size()
           << " input operand(s), but the regions consume only " << inputIdx;

  return verifyInputResidency(*this, getInputs());
}

void DeviceGraphCollectiveCreateOp::build(
    OpBuilder &builder, OperationState &result, Value theContext,
    llvm::ArrayRef<Value> deviceContexts, Value cache,
    llvm::ArrayRef<Value> inputs, llvm::ArrayRef<unsigned> inputCounts,
    llvm::function_ref<void(OpBuilder &, Location, unsigned, Value, Value,
                            ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  unsigned numRegions = inputCounts.size();
  unsigned inputTotal = 0;
  for (unsigned count : inputCounts)
    inputTotal += count;
  assert(inputTotal == inputs.size() &&
         "inputCounts must sum to inputs.size()");

  MLIRContext *ctx = builder.getContext();
  SmallVector<Type> graphTypes;
  graphTypes.reserve(numRegions);

  size_t inputOffset = 0;

  for (unsigned i : llvm::seq(numRegions)) {
    Region *region = result.addRegion();
    Block *block = builder.createBlock(region);

    Value graphBuilder =
        block->addArgument(DeviceGraphBuilderType::get(ctx), result.location);
    Value contextArg =
        block->addArgument(theContext.getType(), result.location);

    unsigned count = inputCounts[i];
    ValueRange argSlice =
        llvm::ArrayRef<Value>(inputs).slice(inputOffset, count);
    SmallVector<Location> locs(count, result.location);
    auto inputArgs = block->addArguments(argSlice.getTypes(), locs);

    if (bodyBuilder)
      bodyBuilder(builder, result.location, i, graphBuilder, contextArg,
                  ValueRange(inputArgs));

    if (!block->mightHaveTerminator())
      ensureTerminator(*region, builder, result.location);

    auto yieldOp = cast<DeviceGraphYieldOp>(block->getTerminator());
    FunctionType signature = FunctionType::get(ctx, argSlice.getTypes(),
                                               yieldOp.getOutputs().getTypes());
    graphTypes.push_back(DeviceGraphType::get(ctx, signature));

    inputOffset += count;
  }

  build(builder, result, graphTypes, theContext, deviceContexts, cache, inputs,
        /*stableTypes=*/StableAddressArrayAttr(),
        /*shapeOnlyInputs=*/mlir::DenseI64ArrayAttr(), /*graphsCount=*/0);
}

/// Emits the build closure, shared by the cached and uncached paths, that
/// dispatches to the body of region `n` for a `DeviceGraphBuilder`.
static FailureOr<MOGG::MojoValue>
emitCollectiveBuildForClosure(MOGG::EmitMojoState &state,
                              DeviceGraphCollectiveCreateOp op) {
  using namespace MOGG;

  NamedMojoDef nParam("n", MojoValue("Int"));
  NamedMojoDef builderParam =
      NamedMojoDef::asVerbatim("builder", MojoValue(kStructDeviceGraphBuilder));
  builderParam.argConvention = "mut";

  MojoValue sharedContext = state.getMojoValueFor(op.getTheContext());

  llvm::SmallBitVector shapeOnlyInputs = op.shapeOnlyInputs();

  bool errOccurred = false;
  MojoValue buildFor = state.builder.lambda(
      "collective_build_for", /*parameters=*/{nParam},
      /*arguments=*/{builderParam},
      /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true,
      /*captureConvention=*/"imm", [&](const FunctionBody &buildBody) {
        MojoValue n = buildBody.parameters.front();
        MojoValue builderArgMojo = buildBody.arguments.front();

        for (Region &region : op.getGraphs()) {
          state.setMojoValueFor(region.front().getArgument(1), sharedContext);
          state.setMojoValueFor(region.front().getArgument(0), builderArgMojo);
        }

        size_t inputOffset = 0;
        MojoBuilder *branch = nullptr;
        for (auto [idx, region] : llvm::enumerate(op.getGraphs())) {
          Block &block = region.front();
          size_t numRegionInputs = block.getNumArguments() - 2;
          auto regionInputs =
              op.getInputs().slice(inputOffset, numRegionInputs);

          SmallVector<Value> blockInputs(block.getArguments().begin() + 2,
                                         block.getArguments().end());

          auto regionBody = [&] {
            DeviceGraphBuilderRef builderValue(builderArgMojo);
            for (auto [i, arg, operand] :
                 llvm::enumerate(blockInputs, regionInputs)) {
              if (!isDeviceGraphInput(operand)) {
                state.setMojoValueFor(arg, state.getMojoValueFor(operand));
                continue;
              }
              MojoValue input = state.getMojoValueFor(operand);
              // The tensor was unpacked once with the kind its type implies;
              // a region that reads only its shape rebinds it so `add_input`
              // records it in place instead of copying it.
              if (shapeOnlyInputs.test(inputOffset + i)) {
                MojoValue kind = MojoValue(kStructDeviceGraphInputKind)
                                     .get(kInputKindShapeOnly);
                input = state.builder.var(input.get(kFnWithKind)(
                    /*parameters=*/{kind}, /*operands=*/{}));
              }
              MojoValue stable =
                  state.builder.var(builderValue.addInput(input));
              state.setMojoValueFor(arg, stable);
            }

            for (Operation &opInBody : block.getOperations())
              if (state.lowerOpToMojo(&opInBody).failed()) {
                errOccurred = true;
                return;
              }
          };

          if (idx == 0)
            branch = &state.builder.if_(n.eq(MojoValue(idx)), regionBody,
                                        /*comptime=*/true);
          else
            branch = &branch->elif_(n.eq(MojoValue(idx)), regionBody);

          inputOffset += numRegionInputs;
        }

        branch->else_([&] {
          state.builder.raise(
              "Error(\"unexpected device index in collective device graph "
              "build\")");
        });
      });

  if (errOccurred)
    return failure();
  return buildFor;
}

/// Emits the per-device cache-key closure from that region's inputs only;
/// `DeviceGraph.create_collective` adds the closure identity separately.
static MOGG::MojoValue
emitCollectiveKeyForClosure(MOGG::EmitMojoState &state,
                            DeviceGraphCollectiveCreateOp op) {
  using namespace MOGG;

  NamedMojoDef nParam("n", MojoValue("Int"));

  return state.builder.lambda(
      "collective_key_for", /*parameters=*/{nParam},
      /*arguments=*/{},
      /*returnType=*/MojoValue(kStructString),
      MojoBuilder::LambdaInline::kNever,
      /*isAsync=*/false, /*raises=*/true, /*captureConvention=*/"imm",
      [&](const FunctionBody &keyBody) {
        MojoValue n = keyBody.parameters.front();

        size_t inputOffset = 0;
        MojoBuilder *branch = nullptr;
        for (auto [idx, region] : llvm::enumerate(op.getGraphs())) {
          Block &block = region.front();
          size_t numRegionInputs = block.getNumArguments() - 2;
          auto regionInputs =
              op.getInputs().slice(inputOffset, numRegionInputs);

          auto keyRegionBody = [&] {
            SmallVector<MojoValue> keyInputs;
            for (Value input : regionInputs)
              if (isDeviceGraphInput(input))
                keyInputs.push_back(state.getMojoValueFor(input));

            MojoValue key = state.builder.var(
                MojoValue(kStructDeviceGraphCache)
                    .get("make_key")(/*parameters=*/{}, keyInputs));
            state.builder.returnsValue(key.transfer());
          };

          if (idx == 0)
            branch = &state.builder.if_(n.eq(MojoValue(idx)), keyRegionBody,
                                        /*comptime=*/true);
          else
            branch = &branch->elif_(n.eq(MojoValue(idx)), keyRegionBody);

          inputOffset += numRegionInputs;
        }

        branch->else_([&] {
          state.builder.raise(
              "Error(\"unexpected device index in collective device graph "
              "key\")");
        });
      });
}

llvm::SmallBitVector DeviceGraphCollectiveCreateOp::shapeOnlyInputs() {
  llvm::SmallBitVector result(getInputs().size());

  if (std::optional<ArrayRef<int64_t>> attr = getShapeOnlyInputs()) {
    for (int64_t idx : attr.value())
      result.set(idx);
  }

  return result;
}

LogicalResult
DeviceGraphCollectiveCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  FailureOr<MojoValue> buildFor = emitCollectiveBuildForClosure(state, *this);
  if (failed(buildFor))
    return failure();

  unsigned numGraphs = getGraphs().size();

  MojoValue contextsArrayType =
      makeMojoArrayType(kStructDeviceContext, numGraphs);
  MojoValue graphsArrayType = makeMojoArrayType(kStructDeviceGraph, numGraphs);

  MojoValue graphsArray;
  if (Value cache = getCache()) {
    MojoValue keyFor = emitCollectiveKeyForClosure(state, *this);

    SmallVector<MojoValue> deviceContextValues =
        state.getMojoValuesFor(getDeviceContexts());
    MojoValue contextsArray = state.builder.varDecl(contextsArrayType);
    state.builder.assign(contextsArray,
                         state.builder.listLiteral(deviceContextValues));

    SmallVector<MojoValue> collectiveArgs = {
        contextsArray, *buildFor, keyFor,
        DeviceGraphRef::cacheArg(state.getMojoValueFor(cache))};

    // Each stable address slot is allocated on the device its type names;
    // the Mojo side takes that as an ordinal into the contexts array.
    if (StableAddressArrayAttr stableTypes = getStableTypesAttr();
        stableTypes && !stableTypes.getValue().empty()) {
      SmallVector<MojoValue> stableDevices;
      for (auto [slot, type] : llvm::enumerate(stableTypes.getValue())) {
        DeviceRefAttr device = type.getElementType().getDevice();
        auto contexts = getDeviceContexts();
        auto it = llvm::find_if(contexts, [&](Value ctx) {
          return cast<DeviceContextType>(ctx.getType()).getDevice() == device;
        });
        if (!device || it == contexts.end())
          return emitOpError("stable slot #")
                 << slot << " lives on device " << device
                 << ", which is not one of the collective's device contexts";
        stableDevices.push_back(
            MojoValue(static_cast<uint64_t>(it - contexts.begin())));
      }
      collectiveArgs.push_back(DeviceGraphRef::stableDevicesArg(
          state.builder.listLiteral(stableDevices)));
    }

    graphsArray =
        state.builder.var(DeviceGraphRef::createCollective(collectiveArgs));
  } else {
    if (StableAddressArrayAttr stableTypes = getStableTypesAttr();
        stableTypes && !stableTypes.getValue().empty())
      return emitOpError("stable address slots require a device graph cache "
                         "to own them; the op has no `cache` operand");

    // Uncached path: build each graph individually so the op does not require
    // a shared cache.
    NamedMojoDef builderParam = NamedMojoDef::asVerbatim(
        "builder", MojoValue(kStructDeviceGraphBuilder));
    builderParam.argConvention = "mut";

    graphsArray = state.builder.varDecl(MojoValue(graphsArrayType));
    for (auto [idx, ctx] : llvm::enumerate(getDeviceContexts())) {
      MojoValue buildDevice = state.builder.lambda(
          "collective_build_device", /*parameters=*/{},
          /*arguments=*/{builderParam},
          /*returnType=*/std::nullopt, MojoBuilder::LambdaInline::kNever,
          /*isAsync=*/false, /*raises=*/true,
          /*captureConvention=*/"var", [&](const FunctionBody &bd) {
            state.builder.call(*buildFor, /*parameters=*/{MojoValue(idx)},
                               /*operands=*/{bd.arguments.front()});
          });

      MojoValue graph = state.builder.var(
          DeviceGraphRef::create({state.getMojoValueFor(ctx), buildDevice}));
      state.builder.assign(graphsArray[MojoValue(idx)], graph);
    }
  }

  // Named per-device variables keep an origin so consumers can transfer them
  // with `^`.
  for (unsigned i = 0; i < numGraphs; ++i) {
    MojoValue graphVar = state.builder.var(graphsArray[MojoValue(i)]);
    state.setMojoValueFor(getOutGraphs()[i], graphVar);
  }

  return success();
}

bool DeviceGraphCollectiveCreateOp::isThrows() { return true; }

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphCollectiveExecuteOp::verify() {
  size_t expectedInputCount = 0;
  size_t expectedOutputCount = 0;

  for (auto graph : getGraphs()) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();
    expectedInputCount += sig.getInputs().size();
    expectedOutputCount += sig.getResults().size();
  }

  if (getInputs().size() != expectedInputCount)
    return emitOpError("`inputs` has ")
           << getInputs().size()
           << " operands, but the graph signatures consume "
           << expectedInputCount;

  if (getOutputs().size() != expectedOutputCount)
    return emitOpError("`outputs` has ")
           << getOutputs().size()
           << " results, but the graph signatures produce "
           << expectedOutputCount;

  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  SmallVector<Type> outputTypes;
  llvm::append_range(outputTypes, getOutputs().getTypes());

  size_t inputIdx = 0;
  size_t outputIdx = 0;
  for (auto [graphIdx, graph] : llvm::enumerate(getGraphs())) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();

    if (!llvm::equal(
            sig.getInputs(),
            ArrayRef<Type>(inputTypes).slice(inputIdx, sig.getInputs().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature inputs do not match the corresponding `inputs` "
                "operands";
    inputIdx += sig.getInputs().size();

    if (!llvm::equal(sig.getResults(),
                     ArrayRef<Type>(outputTypes)
                         .slice(outputIdx, sig.getResults().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature results do not match the corresponding `outputs` "
                "results";
    outputIdx += sig.getResults().size();
  }

  return verifyInputResidency(*this, getInputs());
}
