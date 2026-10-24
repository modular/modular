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

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "MLRT/include/MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/include/MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "MLRT/include/MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace M::MGP {

#define GEN_PASS_DEF_OUTLINEMODELSTOCALLABLES
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

using namespace M;
using namespace M::MGP;

namespace {

class OutlineModelsToCallablesPass
    : public impl::OutlineModelsToCallablesBase<OutlineModelsToCallablesPass> {
  using OutlineModelsToCallablesBase::OutlineModelsToCallablesBase;

  struct GraphContext;

  /// Corresponds 1:1 with `MGP::ModelOp`s, but outlives them to pass context
  /// to a subgraph's parent during `rewriteAndSeal()`.
  struct GraphContext {
    struct SealedGraphContext {
      MEF::CallableOp initCallable;
      MEF::CallableOp execCallable;

      SealedGraphContext(MEF::CallableOp init, MEF::CallableOp exec)
          : initCallable(init), execCallable(exec) {}
    };

    /// Only exists once `sealed` is true
    std::optional<SealedGraphContext> sealedContext;

    /// Only exists while `sealed` is false
    std::optional<MGP::ModelOp> unsealedContext;

    std::optional<mlir::Value> initPrefixValue;
    std::optional<mlir::Value> execPrefixValue;

    bool hasSubgraphs;

    llvm::SmallVector<GraphContext *> referencedBy;

    GraphContext(MGP::ModelOp model)
        : sealedContext(), unsealedContext(model), hasSubgraphs(false),
          sealed(false) {}

    /// GraphContexts reference each other, so copying is ill-defined.
    GraphContext(const GraphContext &) = delete;
    GraphContext &operator=(const GraphContext &) = delete;

    /// Movable so it can be placed into a unique_ptr at creation.
    GraphContext(GraphContext &&) = default;

    /// Marks this context rewritten; the original `MGP::ModelOp` is gone, so
    /// later reads go through `sealedContext` and the other fields.
    void seal() {
      ASSERT_STREAM(this->sealedContext.has_value(),
                    "can't seal a context without initializing sealedContext");
      ASSERT_STREAM(!this->sealed,
                    "tried to seal a GraphContext that was already sealed");
      this->sealed = true;
    }

    /// See `seal()`
    bool isSealed() { return this->sealed; }

  private:
    /// See `seal()`
    bool sealed;
  };

  /// Per-invocation state: the module's call graph of `GraphContext`s.
  struct ModuleContext {
    ModuleOp moduleOp;

    llvm::DenseMap<StringAttr, std::unique_ptr<GraphContext>> models;

    /// Non-subgraph models; a GraphRT-executable module has at most one.
    llvm::SmallVector<GraphContext *, 1> entryGraphs;

    GraphContext *intern(GraphContext gc) {
      ASSERT_STREAM(!gc.isSealed(),
                    "tried to intern an already-sealed GraphContext");

      auto gcUnique = std::make_unique<GraphContext>(std::move(gc));
      auto gcPtr = gcUnique.get();

      if (!gc.unsealedContext.value().getIsSubgraph())
        this->entryGraphs.push_back(gcPtr);

      bool inserted = this->models
                          .insert({gcPtr->unsealedContext.value().getNameAttr(),
                                   std::move(gcUnique)})
                          .second;

      ASSERT_STREAM(inserted, "two models with the same symbol name existed:"
                                  << gcPtr->unsealedContext.value().getName());

      return gcPtr;
    }

    GraphContext *getModel(StringAttr name) {
      auto model = this->models.find(name);
      if (model != this->models.end())
        return model->getSecond().get();
      else
        return nullptr;
    }

    ModuleContext(ModuleOp moduleOp) : moduleOp(moduleOp), models() {
      for (auto model : moduleOp.getOps<MGP::ModelOp>())
        this->intern(GraphContext(model));

      for (auto &[name, gc] : this->models) {
        gc->unsealedContext.value().getExecute().walk([&](MGP::CallOp call) {
          auto &callee = *this->models.at(call.getCalleeAttr().getAttr());
          gc->hasSubgraphs = true;
          callee.referencedBy.push_back(&*gc);
        });
      }
    }
  };

  /// Pre-sealing only. Adds a prefix arg to each subgraph's init and exec,
  /// filled in by rewriteAndSeal, and concatenates it onto placeholder
  /// BufferConstantExternalOp weight names at runtime.
  void localPropagatePrefixes(ModuleContext &context,
                              std::optional<mlir::Value> prefix) {
    for (auto &[name, modelPtr] : context.models) {
      ModelOp model = modelPtr.get()->unsealedContext.value();
      if (model.getIsSubgraph()) {
        auto builder = mlir::OpBuilder(model.getContext());
        auto implicitArgsCount = model.getExecuteImplicitArguments().size();
        Block *execBlock = model.getExecuteBlock();
        Block *initBlock = model.getInitBlock();

        auto mojoStringType = MGP::OpaqueType::get(
            builder.getContext(), builder.getStringAttr("mojo_string"), {});

        // Exec: between the implicit and explicit args. Init: at the end.
        Value execPrefix = execBlock->insertArgument(
            implicitArgsCount, mojoStringType, model->getLoc());

        modelPtr.get()->execPrefixValue = execPrefix;

        Value initPrefix =
            initBlock->addArgument(mojoStringType, model->getLoc());

        // mgp.call is fixed up later, during conversion.
        llvm::SmallVector<std::pair<Block *, mlir::Value>> blocks(
            {{initBlock, initPrefix}, {execBlock, execPrefix}});

        for (auto [block, prefix] : blocks) {
          block->walk([&](MGP::BufferConstantExternalOp op) {
            if (!op.getEIsPlaceholder()) {
              // Non-placeholders already have their final name.
              return mlir::WalkResult::skip();
            }

            builder.setInsertionPoint(op);

            Value rawName = op.getName();

            Value concatenatedName = StringConcatOp::create(
                builder, op.getLoc(), mojoStringType, prefix, rawName);

            op.getNameMutable().assign(concatenatedName);

            return mlir::WalkResult::advance();
          });
        }

        modelPtr.get()->initPrefixValue = initPrefix;

        // Init returns the prefix so exec need not rebuild it.
        llvm::cast<MEF::OutputOp>(initBlock->getTerminator())
            .getOperandsMutable()
            .append(initPrefix);
      }
    }
  }

  /// Post-order over the model call graph, rejecting cycles. Each unvisited
  /// model is split into init and exec `MEF::CallableOp`s. Each `MGP::CallOp`
  /// becomes a `GRT::Invoke` in init and in exec, with the callee's init
  /// outputs (plus explicit args and prefix) forwarded to its exec; these
  /// become this graph's own forwarded bundle. A non-subgraph model is also
  /// wrapped in a `MEF::ModelOp` GraphRT can execute. The original
  /// `MGP::ModelOp` is then erased and the context sealed.
  FailureOr<GraphContext *>
  rewriteAndSeal(ModuleContext &context, GraphContext &graph,
                 llvm::SmallVector<ModelOp> ancestors) {

    if (graph.isSealed())
      return &graph;

    // Cheap: subgraph call depth rarely exceeds 2.
    for (ModelOp ancestor : ancestors) {
      if (ancestor.getNameAttr() ==
          graph.unsealedContext.value().getNameAttr()) {
        InFlightDiagnostic diag = graph.unsealedContext.value().emitOpError(
            "cycle detected in subgraph call graph");

        // Annotate only the subgraphs that form the cycle, starting from the
        // ancestor we are re-entering. Ancestors before that point are on the
        // call stack but not part of the cycle itself.
        bool inCycle = false;
        for (ModelOp diagAncestor : ancestors) {
          if (diagAncestor.getNameAttr() ==
              graph.unsealedContext.value().getNameAttr())
            inCycle = true;
          if (inCycle)
            diag.attachNote(diagAncestor.getLoc()) << "subgraph in cycle";
        }

        return failure();
      }
    }

    ancestors.push_back(graph.unsealedContext.value());

    ModelOp model = graph.unsealedContext.value();
    Location modelLoc = model->getLoc();
    StringRef modelName = model.getName().getValue();
    bool isSubgraph = model.getIsSubgraph();

    auto initOutput =
        llvm::cast<MEF::OutputOp>(model.getInitBlock()->getTerminator());

    // Collected up front since fixups take several passes.
    struct CallContext {
      MGP::CallOp call;
      StringAttr prefixAttr;
      GraphContext *toGraph;

      // Results of the callee's `init` invoke, forwarded as a bundle.
      llvm::SmallVector<mlir::Value> initResults;
      llvm::SmallVector<Type> initResultTypes;

      llvm::SmallVector<mlir::Value> execArgsBundle;
    };

    llvm::SmallVector<CallContext> calls;

    WalkResult result = model.getExecute().walk([&](MGP::CallOp call) {
      GraphContext *maybeCallee =
          context.getModel(call.getCalleeAttr().getAttr());
      ASSERT_STREAM(maybeCallee, << "CallOp's callee did not exist in module");

      StringAttr prefix = call.getPrefixAttr();

      FailureOr<GraphContext *> sealed =
          rewriteAndSeal(context, *maybeCallee, ancestors);

      if (failed(sealed))
        return WalkResult::interrupt();

      calls.push_back({call, prefix, sealed.value(), {}, {}, {}});
      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return failure();

    // Child init outputs are stacked onto this init's outputs and threaded
    // into this exec's inputs:
    {
      OpBuilder initBuilder(model.getInitBlock()->getTerminator());
      // Init is going to receive:
      //  (...<existing ModelOp init operands>, <prefix>)
      // Init is going to output:
      //  (<existing ModelOp init results>, <prefix>, <inner callee bundles>)
      //
      // This entire thing becomes a single inner callee bundle as viewed from
      // "above"
      //
      // Exec is going to receive:
      //  (<existing ModelOp init results>, <prefix>, <inner callee bundles>,
      //  <explicit arguments>
      //
      // Exec is going to output:
      //  (<ModelOp exec results>)

      SmallVector<Value> initChains;

      for (auto &[call, prefix, ctx, initResults, initResultTypes,
                  execArgsBundle] : calls) {
        GraphContext::SealedGraphContext target = ctx->sealedContext.value();

        // The prefix also travels through the bundle to exec.
        auto mojoStringType =
            MGP::OpaqueType::get(initBuilder.getContext(),
                                 initBuilder.getStringAttr("mojo_string"), {});

        Value prefixValue = StringCreateOp::create(initBuilder, call->getLoc(),
                                                   mojoStringType, prefix)
                                .getResult();
        if (graph.execPrefixValue.has_value()) {
          auto parentPrefix = graph.execPrefixValue.value();
          prefixValue =
              StringConcatOp::create(initBuilder, call->getLoc(),
                                     mojoStringType, parentPrefix, prefixValue)
                  .getResult();
        }

        auto initArgs =
            llvm::SmallVector<Value>(model.getInitBlock()->getArguments());

        initArgs.push_back(prefixValue);

        auto initInvokeOp = GRT::InvokeOp::create(
            initBuilder, call.getLoc(), target.initCallable.getResultTypes(),
            target.initCallable.getName(), initArgs);

        auto results = initInvokeOp.getResults();

        initResults.append(results.begin(), results.end());

        initResultTypes.append(results.getTypes().begin(),
                               results.getTypes().end());

        initChains.push_back(initArgs.front());
      }

      auto initOutputOperands = initOutput.getOperandsMutable();

      for (auto &callContext : calls) {
        size_t insertionIndex = initOutputOperands.size();
        initOutputOperands.append(callContext.initResults);

        for (size_t i = 0; i < callContext.initResultTypes.size(); i++) {
          auto execArg = model.getExecuteBlock()->insertArgument(
              insertionIndex + i, callContext.initResultTypes[i],
              model->getLoc());

          callContext.execArgsBundle.push_back(execArg);
        }
      }

      for (auto &callContext : calls) {
        // Forwarded bundles go between the implicit args (incl. prefix) and
        // the explicit args; -1 keeps the prefix.
        auto numOperands = callContext.call.getOperands().size();
        auto dropCount = callContext.initResults.size() - 1;
        ASSERT_STREAM(numOperands >= dropCount,
                      << "call has " << numOperands
                      << " operands but trying to drop " << dropCount);
        auto explicitArgs =
            callContext.call.getOperands().drop_front(dropCount);

        auto &args = callContext.execArgsBundle;
        args.append(explicitArgs.begin(), explicitArgs.end());

        GraphContext::SealedGraphContext target =
            callContext.toGraph->sealedContext.value();

        OpBuilder execBuilder(callContext.call);

        auto execInvokeOp =
            GRT::InvokeOp::create(execBuilder, callContext.call->getLoc(),
                                  target.execCallable.getResultTypes(),
                                  target.execCallable.getName(), args);

        callContext.call->replaceAllUsesWith(execInvokeOp->getResults());
        callContext.call->erase();
      }

      // The parent init also waits on child inits; some callers implicitly
      // expect that.
      Value parentChain = initOutput.getOperand(0);
      initChains.push_back(parentChain);
      auto newInitChain = GRT::ChainCreateOp::create(
          initBuilder, initOutput->getLoc(), initChains);
      initOutput->setOperand(0, newInitChain.getResult());
    }

    mlir::OpBuilder builder(context.moduleOp);

    auto initFunctionType = mlir::FunctionType::get(
        context.moduleOp.getContext(), model.getInitBlock()->getArgumentTypes(),
        initOutput->getOperandTypes());

    auto execFunctionType = mlir::FunctionType::get(
        context.moduleOp.getContext(),
        model.getExecuteBlock()->getArgumentTypes(),
        model.getExecuteBlock()->getTerminator()->getOperandTypes());

    builder.setInsertionPoint(context.moduleOp);
    auto parent = ancestors.back();
    builder.setInsertionPoint(parent);

    auto initCallable = MEF::CallableOp::create(
        builder, modelLoc,
        builder.getStringAttr(modelName + mlir::Twine("$init")), {}, {}, {});

    initCallable.setFunctionType(initFunctionType);

    auto execCallable = MEF::CallableOp::create(
        builder, modelLoc,
        builder.getStringAttr(modelName + mlir::Twine("$exec")), {}, {}, {});

    execCallable.setFunctionType(execFunctionType);

    initCallable.getCallableRegion()->takeBody(model.getInit());
    execCallable.getCallableRegion()->takeBody(model.getExecute());

    graph.sealedContext =
        GraphContext::SealedGraphContext(initCallable, execCallable);

    if (!isSubgraph) {
      // The runtime needs a wrapper model as its entry point.
      auto loc = model->getLoc();

      mlir::FunctionType initFnType = initCallable.getFunctionType();
      mlir::FunctionType execFnType = execCallable.getFunctionType();

      auto devices = model.getDevices();
      auto frameworkName = model.getFrameworkName();

      // Read before erasing the model.
      auto initialModelFnType = model.getInputOutputTypes();
      auto extraAttrs = model.getTheExtraAttrsAttr();

      OpBuilder builder(model->getContext());
      builder.setInsertionPointAfter(model);

      auto resultDeviceIndicesAttr = builder.getArrayAttr({});
      auto argumentDeviceIndicesAttr = builder.getArrayAttr({});
      auto argumentNamesAttr = builder.getArrayAttr({});
      auto resultNamesAttr = builder.getArrayAttr({});
      if (extraAttrs) {
        auto resultDeviceIndices = extraAttrs.get(kMgpModelResultDeviceIndices);
        if (auto indices =
                llvm::cast_if_present<mlir::ArrayAttr>(resultDeviceIndices))
          resultDeviceIndicesAttr = indices;

        auto argumentDeviceIndices =
            extraAttrs.get(kMgpModelArgumentDeviceIndices);
        if (auto indices =
                llvm::cast_if_present<mlir::ArrayAttr>(argumentDeviceIndices))
          argumentDeviceIndicesAttr = indices;

        if (auto names = llvm::cast_if_present<mlir::ArrayAttr>(
                extraAttrs.get(kMgpModelArgumentNames)))
          argumentNamesAttr = names;

        if (auto names = llvm::cast_if_present<mlir::ArrayAttr>(
                extraAttrs.get(kMgpModelResultNames)))
          resultNamesAttr = names;
      }

      auto mefModel = MEF::ModelOp::create(
          builder, loc, devices, frameworkName, initFnType, execFnType,
          initialModelFnType, argumentDeviceIndicesAttr,
          resultDeviceIndicesAttr, argumentNamesAttr, resultNamesAttr,
          modelName, /*sym_visibility=*/nullptr);

      auto &initBlock = mefModel.getInit().emplaceBlock();
      auto &execBlock = mefModel.getExecute().emplaceBlock();

      SmallVector<std::pair<Block *, MEF::CallableOp>> blocks{
          {&initBlock, initCallable}, {&execBlock, execCallable}};

      for (auto &[block, callable] : blocks) {
        builder.setInsertionPointToEnd(block);

        SmallVector<Value> operands;

        for (auto argTy : callable.getArgumentTypes()) {
          auto arg = block->addArgument(argTy, loc);
          operands.push_back(arg);
        }

        auto invoke =
            GRT::InvokeOp::create(builder, loc, callable.getResultTypes(),
                                  callable.getName(), operands);

        MEF::OutputOp::create(builder, loc, invoke.getResults());
      }
    }

    graph.unsealedContext->erase();
    graph.unsealedContext = {};

    graph.seal();
    ancestors.pop_back();
    return &graph;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    ModuleContext context(moduleOp);

    // Propagating prefixes first saves operand juggling in the rewrite.
    localPropagatePrefixes(context, {});

    for (GraphContext *entryGraph : context.entryGraphs) {
      ASSERT_STREAM(!entryGraph->unsealedContext.value().getIsSubgraph(),
                    "no entry graph should be a subgraph");

      llvm::SmallVector<ModelOp> ancestors;

      FailureOr<GraphContext *> result =
          rewriteAndSeal(context, *entryGraph, ancestors);

      if (failed(result))
        signalPassFailure();
    }
  }
};

} // namespace
} // namespace M::MGP
