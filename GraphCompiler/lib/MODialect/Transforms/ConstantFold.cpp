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

#include "GraphCompiler/MODialect/Transforms/ConstantFold.h"
#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/Transforms/ConstantFoldPass.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Passes.h"
#include "GraphCompiler/Support/BlobLifetimeManager.h"
#include "GraphCompiler/Support/List.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Support/Buffer.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/Context.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace M::MO;

namespace M::MO {
#define GEN_PASS_DEF_CONSTANTFOLD
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"
} // namespace M::MO

namespace {

// Holds all the information necessary to execute and replace all constant
// subgraphs in a single graph op.
struct SubgraphsInfo {
  GraphOp orig;
  std::string constantSubgraphName;
  // Need the subgraph for its output values.
  SmallVector<Subgraph> subgraphs;
};

using RuntimeTensor = std::pair<ElementsAttr, TensorShape>;

struct ModelResult {
  /// Subgraph output op to its replacement constants, with runtime shapes so
  /// unknown dims can be resolved.
  llvm::DenseMap<Operation *, SmallVector<RuntimeTensor>> constants;
  /// Subgraph parameter name to its folded value.
  llvm::DenseMap<StringAttr, int64_t> parameters;
};

// Models need to ensure that their backing buffer lives as long as they do.
struct ModelsWithBuffer {
  ModelList models;
  BufferRef buffer;
};

} // namespace

static const Model *findModelByName(const ModelList &models, StringRef name) {
  for (const std::unique_ptr<const Model> &model : models) {
    if (model && model->getName() == name)
      return model.get();
  }
  return nullptr;
}

llvm::SetVector<StringAttr> M::MO::Subgraph::paramDeclarations() const {
  llvm::SetVector<StringAttr> params;
  for (auto op : ops) {
    op->walk([&](KGEN::ParamOpInterface paramOp) {
      paramOp.walkDeclarations(
          [&](KGEN::ParamDeclAttr decl) { params.insert(decl.getName()); });
      paramOp.walkDefinitions(
          [&](KGEN::ParamDeclAttr decl, const KGEN::ParamDefValue &val) {
            params.insert(decl.getName());
          });
    });
  }
  return params;
}

/// Topologically sort parameters in a list of ops.
static llvm::SmallVector<Operation *>
sortParams(llvm::ArrayRef<Operation *> ops) {
  // ParameterUseDefGraph doesn't map from parameter to operations using it, so
  // we have to construct our own.
  llvm::DenseMap<StringAttr, llvm::SetVector<StringAttr>> uses;
  llvm::DenseMap<StringAttr, Operation *> defs;
  for (auto op : ops) {
    op->walk([&](KGEN::ParamOpInterface paramOp) {
      paramOp.walkDeclarations([&](KGEN::ParamDeclAttr decl) {
        defs[decl.getName()] = op;
        uses[decl.getName()] = {};
      });
      paramOp.walkDefinitions(
          [&](KGEN::ParamDeclAttr decl, const KGEN::ParamDefValue &val) {
            llvm::SetVector<StringAttr> params;
            for (auto expr : val.exprs) {
              expr.walk([&](KGEN::ParamDeclRefAttr ref) {
                params.insert(ref.getName());
              });
            }
            defs[decl.getName()] = op;
            uses[decl.getName()] = params;
          });
    });
  }

  llvm::SetVector<Operation *> sorted;
  llvm::SetVector<StringAttr> inserted;
  llvm::SmallVector<StringAttr> workQueue;
  for (auto [k, _] : uses)
    workQueue.push_back(k);
  while (!workQueue.empty()) {
    StringAttr param = workQueue.back();
    workQueue.pop_back();

    if (inserted.contains(param))
      continue;
    auto notHandled =
        llvm::make_filter_range(uses.at(param), [&](StringAttr param) {
          return !inserted.contains(param);
        });
    if (notHandled.empty()) {
      inserted.insert(param);
      sorted.insert(defs[param]);
      continue;
    }

    workQueue.push_back(param);
    for (auto dependentParam : notHandled)
      workQueue.push_back(dependentParam);
  }

  for (auto op : ops)
    sorted.insert(op);
  return sorted.takeVector();
}

/// Returns false if `op` (an MO or KGEN param op) is definitely not constant
/// foldable.
static bool isMaybeConstantFoldable(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;

  if (isa<MO::TransferOp>(op))
    return false;

  if (isa<MO::ConstantExternalOp>(op))
    return false;

  // Skip ops with parameterized shapes.
  auto findParamDeclRefAttr = [](KGEN::ParamDeclRefAttr paramRef) {
    return WalkResult::interrupt();
  };

  for (auto operandTy : op->getOperandTypes()) {
    if (operandTy.walk(findParamDeclRefAttr).wasInterrupted()) {
      return false;
    }
  }
  for (auto resultTy : op->getResultTypes()) {
    if (resultTy.walk(findParamDeclRefAttr).wasInterrupted()) {
      return false;
    }
  }
  for (auto attr : op->getAttrs()) {
    if (attr.getValue().walk(findParamDeclRefAttr).wasInterrupted()) {
      return false;
    }
  }

  // Skip ops which declare/define parameters (NOTE: they may not use them)
  auto paramOpInterface = dyn_cast<KGEN::ParamOpInterface>(op);
  if (paramOpInterface) {
    bool definesOrDeclaresParams = false;
    paramOpInterface.walkDefinitions(
        [&](KGEN::ParamDeclAttr param, const ParamDefValue &val) {
          definesOrDeclaresParams = true;
        });
    paramOpInterface.walkDeclarations(
        [&](KGEN::ParamDeclAttr param) { definesOrDeclaresParams = true; });
    if (definesOrDeclaresParams)
      return false;
  }

  bool isMOOrMOGG =
      isa_and_nonnull<MO::MODialect, MOGG::MOGGDialect>(op->getDialect());

  bool isSupportedMOOrMOGG =
      isMOOrMOGG &&
      // TODO (#29484): Graphs with mo.list outputs can't be lowered to
      // MGP.
      llvm::all_of(op->getResultTypes(), [](Type ty) {
        return isa<MO::TensorType, MOSH::ShapeType, MO::ListType>(ty);
      });

  bool isNonMOOrMOGGParamOp =
      isa<KGEN::ParamOpInterface, MOSH::ParamToValueOp, KGEN::ParamDeclareOp>(
          op) &&
      !isMOOrMOGG;

  return (isSupportedMOOrMOGG || isNonMOOrMOGGParamOp) &&
         !isa<MOControlOpInterface>(op);
}

/// Wraps a `dense_resource` as a tensor that owns the blob when all of the
/// constant's users are in `subgraph`, and borrows it otherwise.
static Tensor denseResourceAsTensor(AsyncRT::CPUDevice &cpuDevice,
                                    ConstantOp constant,
                                    DenseResourceElementsAttr denseResource,
                                    const Subgraph &subgraph) {
  mlir::AsmResourceBlob *blob = denseResource.getRawHandle().getBlob();
  TensorType ttype = dyn_cast<TensorType>(constant.getType());
  TensorSpec spec(ttype.getShape().getIntDims(), ttype.getDtype());

  llvm::SetVector<Operation *> constantUsers;
  for (Operation *user : constant->getUsers())
    constantUsers.insert(user);

  llvm::SetVector<Operation *> usersOutsideSubgraph(constantUsers);
  usersOutsideSubgraph.set_subtract(subgraph.ops);

  // Frees the blob when the buffer dies instead of with the MLIRContext.
  if (usersOutsideSubgraph.empty() && blob->getDeleter()) {
    auto async = AsyncValue::createReady<GML::BlobLifetimeManager>(
        cpuDevice.getCompactPtr(), std::move(blob->getDeleter()),
        blob->getData(), blob->getDataAlignment());
    blob->getDeleter() = nullptr;
    auto ref = AnyAsyncValueRef::take(async);

    TensorBufferRef blobBuffer = TensorBufferRef::create(
        reinterpret_cast<void *>(const_cast<char *>(blob->getData().data())),
        blob->getData().size(), std::move(ref), blob->getDataAlignment());
    return Tensor(std::move(blobBuffer), std::move(spec));
  }

  return Tensor::createBorrowed(blob->getData().data(), spec,
                                blob->getDataAlignment());
}

/// Creates an `ElementsAttr` with the same data as the tensor in `ref`.
static ElementsAttr elementsAttrFromTensor(const AnyAsyncValueRef &ref,
                                           MLIRContext *ctx) {
  // The size is only known at runtime.
  Tensor &tensor = ref.get<Tensor>();
  TensorShape shape = tensor.getShape();
  size_t numElements = shape.getNumElements();
  ArrayType arrayType = ArrayType::get(
      ctx, numElements, toMLIRType(ctx, tensor.getEltType()).get());
  ArrayRef<char> tensorData(tensor.getBuffer<char>(), tensor.getSizeInBytes());
  size_t elementByteAlign = llvm::PowerOf2Ceil(
      llvm::divideCeil(arrayType.getElementTypeBitWidth(), CHAR_BIT));
  if (!shouldUseOutOfLineAttrStorage(arrayType.getNumElements()))
    return getInlineAttrForTensorDataCopy(arrayType, tensorData);

  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  TensorBufferRef &bufferRef = tensor.getBufferRef();

  // Reuses the data within the BufferRef and extends the lifetime of it.
  mlir::AsmResourceBlob blob =
      mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
          tensorData, elementByteAlign,
          [storage = bufferRef.getMemStorageHandle().copy()](void *, size_t,
                                                             size_t) {
            // No-op that drops the `bufferRef` when the closure is
            // destroyed.
          });

  return DenseResourceElementsAttr::get(
      arrayType, resourceManager.insert("folded_const", std::move(blob)));
}

static RuntimeTensor extractRuntimeOutput(
    const AnyAsyncValueRef &ref, MLIRContext *ctx,
    llvm::DenseMap<M::AsyncRT::AsyncValue *, ElementsAttr> &cachedConstants) {
  if (M::MLRT::Tensor *tensor = dyn_cast<Tensor>(ref)) {
    ElementsAttr constantsArr;

    auto [it, inserted] =
        cachedConstants.try_emplace(ref.getPointer(), nullptr);
    if (!inserted) {
      // Reuse the attribute for a value returned more than once, avoiding
      // duplicate dense_resources. For example, folding:
      //  mo.if(...) {
      //     %1 = some_op(%cst0)
      //  } else {
      //     %1 = some_op(%cst0)
      // }
      // would otherwise generate:
      // mo.if(...) {
      //   %1 = constant { value=<folded_constant_1> }
      // } else {
      //   %1 = constant { value=<folded_constant_2> }
      // }
      constantsArr = it->second;
    } else {
      constantsArr = elementsAttrFromTensor(ref, ctx);
      it->second = constantsArr;
    }
    return RuntimeTensor(constantsArr, tensor->getShape());
  } else {
    ASSERT_STREAM(false, << "Unexpected GML type");
    llvm_unreachable("Unexpected GML type");
  }
}

/// Executes `model` and returns the folded constants and parameters of its
/// subgraphs.
static AsyncRT::ErrorDiagnosticOr<ModelResult>
executeModel(MLIRContext *ctx, const Model *model,
             const SubgraphsInfo &subgraphsInfo) {
  AsyncRT::CPUDevice &cpuDevice = *loadContext(ctx)->get<AsyncRT::CPUDevice>();
  auto executableModelOr = model->setup(cpuDevice);
  if (executableModelOr.isError())
    return executableModelOr.takeError();

  auto executableModel = executableModelOr.takeValue();
  auto setupErrorOrSuccess = executableModel->waitForModelSetup();
  if (setupErrorOrSuccess.isError()) {
    Error error("Model setup failed, not ready for execution"
                ".  Modular internal detail: " +
                Twine(setupErrorOrSuccess.getError()));
    return AsyncRT::EncodedDiagnostic(std::move(error),
                                      model->getLocation().copy());
  }

  SmallVector<AnyAsyncValueRef> inputTensorsInMEFIndexOrder;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      if (auto denseResource =
              dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr())) {
        inputTensorsInMEFIndexOrder.push_back(
            AsyncValueRef<Tensor>::createReady(
                cpuDevice, denseResourceAsTensor(cpuDevice, constant,
                                                 denseResource, subgraph)));
      }
    }
  }
  auto results = executableModel->execute(inputTensorsInMEFIndexOrder);

  if (results.empty())
    return AsyncRT::EncodedDiagnostic("Failed to execute model",
                                      model->getLocation().copy());

  AsyncRT::await(results);
  for (auto &result : results)
    if (result.isError())
      return AsyncRT::EncodedDiagnostic(result.takeDiagnostic());

  MutableArrayRef<AnyAsyncValueRef> resultCopy(results.data(), results.size());

  [[maybe_unused]] const AnyAsyncValueRef &ch = resultCopy.front();
  ASSERT_STREAM(ch.isReady() && isa<Chain>(ch),
                << "first result must be a ready chain");
  resultCopy = resultCopy.drop_front();

  llvm::DenseMap<M::AsyncRT::AsyncValue *, ElementsAttr> cachedConstants;
  llvm::DenseMap<Operation *, SmallVector<RuntimeTensor>> opToTensors;
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);
  size_t outIdx = 0;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto op : subgraph.outputs) {
      SmallVector<RuntimeTensor> opTensors;
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        AnyAsyncValueRef &tensorRef = resultCopy[outIdx];
        if (tensorRef.isError())
          return AsyncRT::EncodedDiagnostic(tensorRef.takeDiagnostic());

        ASSERT_STREAM(tensorRef.isReady(),
                      << "We already awaited, tensor should be ready");
        opTensors.push_back(
            extractRuntimeOutput(tensorRef, ctx, cachedConstants));
        ++outIdx;
      }
      opToTensors[op] = opTensors;
    }
  }

  // Parameters follow the constants.
  llvm::DenseMap<StringAttr, int64_t> params;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs) {
    for (auto paramName : subgraph.paramDeclarations()) {
      auto &tensor = resultCopy[outIdx];
      if (tensor.isError())
        return AsyncRT::EncodedDiagnostic(tensor.takeDiagnostic());
      ASSERT_STREAM(tensor.isReady(),
                    << "We already awaited, tensor should be ready");
      ASSERT_STREAM(tensor.isType<Tensor>(),
                    << "Subgraph shape output should be a tensor");
      Tensor &t = tensor.get<Tensor>();
      params[paramName] = t.getBuffer<const int64_t>()[0];
      ++outIdx;
    }
  }
  return ModelResult{opToTensors, params};
}

SmallVector<Subgraph>
M::MO::constantSubgraphs(MO::GraphOp graph,
                         KGEN::ParameterCollector::Analysis &paramCache) {
  // Grows constant subgraphs from mo.constant users: an op is constant when
  // all its inputs are, resolving unvisited inputs first. Tracks ops rather
  // than values so parameters are covered too.

  llvm::DenseSet<Operation *> nonConst;

  // Constant ops (mo.constant, or all inputs constant); each class is a
  // connected DAG.
  llvm::EquivalenceClasses<Operation *> constSubgraphs;
  // An op is re-pushed until its operands are resolved.
  llvm::SmallVector<Operation *> workQueue;
  graph.walk([&](ConstantOp op) {
    auto dense = dyn_cast<DenseResourceElementsAttr>(op.getValueAttr());
    // Skip a dense_resource with no blob behind it.
    if (dense && !(dense.getRawHandle().getBlob())) {
      nonConst.insert(op);
      return;
    }
    constSubgraphs.insert(op);
    for (auto user : op.getResult().getUsers())
      workQueue.push_back(user);
  });

  // Bounded to catch unexpected infinite recursion.
  size_t iter = 0;
  size_t boundForMaximumIterations = 1000000;
  for (; iter < boundForMaximumIterations; ++iter) {
    if (workQueue.empty())
      break;

    Operation *op = workQueue.back();
    workQueue.pop_back();

    if (constSubgraphs.contains(op))
      continue;

    if (nonConst.contains(op))
      continue;

    if (!isMaybeConstantFoldable(op)) {
      nonConst.insert(op);
      continue;
    }

    llvm::DenseSet<Operation *> unvisitedOperands;
    bool foundNonConst = false;
    for (auto operand : op->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || nonConst.contains(definition)) {
        foundNonConst = true;
        break;
      }
      if (!constSubgraphs.contains(definition)) {
        unvisitedOperands.insert(definition);
      }
    }
    if (foundNonConst) {
      nonConst.insert(op);
      continue;
    }

    // Revisit this op after its unvisited operands.
    if (!unvisitedOperands.empty()) {
      workQueue.push_back(op);
      for (auto definition : unvisitedOperands)
        workQueue.push_back(definition);
      continue;
    }

    constSubgraphs.insert(op);
    for (auto operand : op->getOperands())
      constSubgraphs.unionSets(op, operand.getDefiningOp());

    for (auto result : op->getResults())
      for (auto user : result.getUsers())
        workQueue.push_back(user);
  }
  ASSERT_STREAM(
      iter < boundForMaximumIterations,
      << "Hit bound on maximum number of iterations for detecting constant "
         "subgraphs. Either this graph op is VERY large (in which case this "
         "bound should be increased), or else somehow there is infinite "
         "recursion (which is a bug)");

  SmallVector<Subgraph> subgraphs;
  for (auto it = constSubgraphs.begin(), end = constSubgraphs.end(); it != end;
       ++it) {
    if (!(*it)->isLeader())
      continue;

    llvm::SetVector<Operation *> members(constSubgraphs.member_begin(*(*it)),
                                         constSubgraphs.member_end());
    Subgraph subgraph(members);

    // Ignore subgraphs that only contain constant ops.
    if (llvm::all_of(subgraph.ops, [](Operation *op) {
          return isa<ConstantOp>(op) || isa<KGEN::ParamDeclareOp>(op);
        }))
      continue;

    subgraphs.push_back(subgraph);
  }

  return subgraphs;
}

/// Clones `subgraphs` into a new GraphOp in `module` that outputs their
/// results. Large constants become inputs of the new graph.
static SubgraphsInfo cloneSubgraphs(OpBuilder builder, GraphOp orig,
                                    ModuleOp module,
                                    const SmallVector<Subgraph> &subgraphs,
                                    int64_t &name_counter) {
  llvm::SmallVector<Type> argTypes;
  for (const Subgraph &subgraph : subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      auto dense = dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr());
      if (dense)
        argTypes.push_back(constant.getType());
    }
  }
  llvm::SmallVector<Type> resultTypes;
  for (const Subgraph &subgraph : subgraphs) {
    for (auto op : subgraph.outputs) {
      for (auto result : op->getResults()) {
        resultTypes.push_back(result.getType());
      }
    }
  }

  // This name cannot be too long otherwise something breaks :(.
  auto graphop = GraphOp::create(builder, module.getLoc(),
                                 (orig.getName() + "_constant_subgraphs").str(),
                                 argTypes, resultTypes);
  builder.setInsertionPointToEnd(graphop.getBodyBlock());

  llvm::MapVector<ConstantOp, BlockArgument> inputConstants;
  int64_t idx = 0;
  for (const Subgraph &subgraph : subgraphs) {
    for (ConstantOp constant : subgraph.constants()) {
      auto dense = dyn_cast<DenseResourceElementsAttr>(constant.getValueAttr());
      if (dense) {
        inputConstants[constant] = graphop.getBody().getArgument(idx);
        ++idx;
      }
    }
  }

  SmallVector<Operation *, 8> opsSorted;
  for (const Subgraph &subgraph : subgraphs)
    for (auto op : subgraph.ops)
      opsSorted.push_back(op);
  // Param sorting alone doesn't keep definitions before uses.
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();
  opsSorted = sortParams(opsSorted);
  mlir::computeTopologicalSorting(opsSorted);
  IRMapping oldToNew;
  for (auto oldOp : opsSorted) {
    ConstantOp oldConst = dyn_cast<ConstantOp>(oldOp);
    if (oldConst) {
      auto it = inputConstants.find(oldConst);
      if (it != inputConstants.end()) {
        oldToNew.map(oldOp->getResult(0), it->second);
        continue;
      }
    }
    builder.clone(*oldOp, oldToNew);
  }
  llvm::SmallVector<Value> newOutputs;
  llvm::SetVector<KGEN::ParamDeclAttr> outputParams;
  for (const Subgraph &subgraph : subgraphs) {
    for (auto op : subgraph.outputs) {
      for (auto result : op->getResults()) {
        Value newOutput = oldToNew.lookup<Value>(result);
        if (llvm::isa<MOSH::ShapeType>(newOutput.getType())) {
          // Shapes can't be outputs; emit a rank-1 tensor of unknown size.
          newOutput = Shape::ToTensorOp::create(
              builder, loc,
              TensorType::get(
                  MOSH::ShapeAttr::get(
                      SmallVector<TypedAttr>(
                          {KGEN::UnknownAttr::get(builder.getIndexType())}),
                      MOSH::ShapeType::get(ctx)),
                  DType::si64),
              newOutput);
        }
        newOutput.getType().walk([&](KGEN::ParamDeclRefAttr ref) {
          outputParams.insert(KGEN::ParamDeclAttr::get(ref));
        });
        newOutputs.push_back(newOutput);
      }
    }
  }

  // Expose each subgraph parameter as a rank-0 int tensor output.
  Type dimTy = MOSH::ShapeAttr::getDimType(ctx);
  for (const Subgraph &subgraph : subgraphs) {
    for (auto param : subgraph.paramDeclarations()) {
      Value dimVal = MOSH::ParamToValueOp::create(
          builder, loc, KGEN::ParamDeclRefAttr::get(ctx, param, dimTy));
      newOutputs.push_back(IndexToTensorOp::create(
          builder, loc,
          TensorType::get(ArrayRef<int64_t>({}),
                          IntegerType::get(ctx, 64, IntegerType::Unsigned)),
          dimVal));
    }
  }
  // No output parameters because we bind them all locally.
  auto out = OutputOp::create(builder, loc, newOutputs,
                              KGEN::ParameterExprArrayAttr{});
  auto functionType =
      mlir::FunctionType::get(ctx, argTypes, out.getOperandTypes());
  graphop.setSignature(
      FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, functionType));
  graphop.setFunctionType(functionType);

  // Rewrite all device attrs to target CPU.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DeviceRefAttr ref) { return DeviceRefAttr::get(ctx, kCPULabel, 0); });
  replacer.recursivelyReplaceElementsIn(graphop, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  return {orig, std::string(graphop.getName()), subgraphs};
}

static Value runtimeTensorToConstant(OpBuilder &builder, Location loc,
                                     RuntimeTensor tensor, Type type) {
  auto [constant, shape] = std::move(tensor);
  return ConstantOp::create(builder, loc, type, constant);
}

/// Replaces subgraph outputs with the executed constants, erases internal ops,
/// and declares the parameters the subgraph defined.
static void replaceConstantOps(GraphOp graph, OpBuilder &builder,
                               const ModelResult &modelResult,
                               const SubgraphsInfo &subgraphsInfo) {
  MLIRContext *ctx = graph.getContext();
  // Collect ops unused outside the subgraph before anything is erased.
  SmallVector<Operation *> allOps;
  for (const Subgraph &subgraph : subgraphsInfo.subgraphs)
    for (auto op : subgraph.ops)
      allOps.push_back(op);
  auto filtered = llvm::make_filter_range(allOps, [&](Operation *op) {
    return llvm::all_of(op->getResults(), [&](OpResult r) {
      return llvm::all_of(
          subgraphsInfo.subgraphs,
          [&](const Subgraph &subgraph) { return subgraph.fullyContains(r); });
    });
  });
  llvm::SmallVector<Operation *> internalOps(filtered.begin(), filtered.end());
  mlir::computeTopologicalSorting(internalOps);

  for (auto constant : modelResult.constants) {
    auto &op = constant.first;
    auto &tensors = constant.second;
    builder.setInsertionPoint(op);
    llvm::SmallVector<Value> newResults;
    for (auto [opResult, constantTensor] :
         llvm::zip(op->getResults(), tensors)) {
      Type type = opResult.getType();
      if (llvm::isa<TensorType>(type)) {
        newResults.push_back(runtimeTensorToConstant(builder, op->getLoc(),
                                                     constantTensor, type));
      } else if (llvm::isa<ShapeType>(type)) {
        auto intElems = dyn_cast<IntArrayElementsAttr>(constantTensor.first);
        newResults.push_back(MOSH::ParamToValueOp::create(
            builder, op->getLoc(),
            ShapeAttr::get(intElems.asArrayRef<int64_t>(),
                           MOSH::ShapeType::get(ctx))));
      } else {
        ASSERT_STREAM(false,
                      << "constants must be tensors or shapes but one is a "
                      << opResult.getType());
      }
    }
    op->replaceAllUsesWith(newResults);
    op->erase();
  }
  // Erase users before their operands.
  for (auto op : llvm::reverse(internalOps))
    op->erase();

  Type dimTy = MOSH::ShapeAttr::getDimType(ctx);
  builder.setInsertionPointToStart(graph.getBodyBlock());
  for (auto [name, val] : modelResult.parameters) {
    KGEN::ParamDeclareOp::create(builder, graph.getLoc(),
                                 KGEN::ParamDeclAttr::get(ctx, name, dimTy),
                                 MOSH::ShapeAttr::getDim(ctx, val));
  }
}

namespace M::MO {

struct ConstantFoldPass
    : public M::MO::impl::ConstantFoldBase<ConstantFoldPass> {
  using ConstantFoldBase::ConstantFoldBase;

  void runOnOperation() override;

  ConstantFoldPass()
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)) {}

  ConstantFoldPass(std::string_view label,
                   std::optional<std::string> cacheDirArg,
                   llvm::ArrayRef<std::string> kernelLibraryPathsArg)
      : ConstantFoldBase(), policy(new FactorMemIncrease(2, 100)),
        compiledFrameworkLabel(label), cacheDir(std::move(cacheDirArg)),
        kernelLibraryPaths(kernelLibraryPathsArg.vec()) {}

private:
  // Lowers `module` to MGP and MEF, then loads its models along with the
  // buffer that must outlive them. A member so it can call runPipeline.
  ErrorOr<ModelsWithBuffer> lowerSubgraphs(ModuleOp module);

  std::shared_ptr<const ConstantFoldPolicy> policy;
  // Label used to load and handle buffers created by the framework.
  std::string compiledFrameworkLabel;
  std::optional<std::string> cacheDir;
  std::optional<std::vector<std::string>> kernelLibraryPaths;
};

ErrorOr<ModelsWithBuffer> ConstantFoldPass::lowerSubgraphs(ModuleOp module) {
  // Seed the sub-module with the parent's kernel decls so the sub-pipeline
  // skips re-importing the `kernels` / `primitives` packages. A standalone
  // `mo-opt` run has no decls, so the sub-pipeline imports them itself.
  bool parentHasDecls =
      (*getOperation()).hasAttr(MOGG::kOpaqueTypeMappingAttrName);
  if (parentHasDecls)
    MOGG::copyKernelDecls(getOperation(), module);

  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  MO::GCPipelineOptions moOpts;
  if (cacheDir)
    moOpts.cacheDir = cacheDir.value();
  if (kernelLibraryPaths)
    moOpts.kernelLibraryPaths = kernelLibraryPaths.value();
  // `copyKernelDecls` already provided the decls and opaque-type mapping.
  moOpts.kernelDeclsPrePopulated = parentHasDecls;
  // Avoid infinite recursion.
  moOpts.constantFold = false;
  // Reduce Memory overhead by not fusing compiled-regions primitives.
  moOpts.enableCompiledRegionsFusion = false;
  // Use MOF executor so we don't depend on a framework.
  moOpts.compiledFrameworkLabel = "mof";
  // This will fold sequence of trivial kernels not on host, leading to huge
  // kernels (in terms of code size).
  moOpts.performTrivialKernelOpt = false;
  // Assign ops to CPU so we can constant fold there.
  moOpts.defaultDeviceLabel = kCPULabel;
  // The folded model runs in this process, so compile for the host even when
  // a virtual CPU target retargets the outer compile (e.g. a Windows-ABI
  // object fails JIT-linking `__chkstk`).
  moOpts.compileForHostExecution = true;
  // Force all kernels to be synchronous. This prevents higher memory usage.
  moOpts.forceSync = true;
  // Constant subgraphs are mostly isolated, so post-ABI fusion only costs
  // compile time.
  moOpts.enablePostABIRegionFusion = false;
  MO::buildGCPipeline(pm, moOpts);
  if (failed(runPipeline(pm, module)))
    return Error("Failed to run MOToMGP pipeline");

  // `convertMGPToMEF` destroys `dense_resource`s; this is safe only because
  // folded subgraphs take them as operands, not attributes.
  auto bufOr = MO::convertMGPToMEF(module, false);
  if (bufOr)
    return bufOr.takeError();
  BufferRef buf = bufOr.takeValue();

  auto mefFileOr = MEFFile::open(buf->getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();
  auto modelsOr = M::loadMGPMEFFile(std::move(*mefFileOr));
  if (modelsOr.isError())
    return modelsOr.takeError();
  return {ModelsWithBuffer{modelsOr.takeValue(), std::move(buf)}};
}

void ConstantFoldPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  for (auto _ : llvm::seq(iterations.getValue())) {
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(getOperation().getBody());
    // Passes can't create ops outside their operation, so subgraph copies
    // live in a temporary nested module.
    OwningOpRef<ModuleOp> module = ModuleOp::create(
        builder, getOperation().getLoc(), "constant_fold_subgraphs");

    // Alternative: merge all subgraphs into one graph and execute it once.
    SmallVector<SubgraphsInfo> subgraphReplacements;
    int64_t nameCounter = 0;

    for (auto graph : getOperation().getOps<GraphOp>()) {
      auto subgraphs = M::MO::constantSubgraphs(
          graph, getAnalysis<KGEN::ParameterCollector::Analysis>());
      if (subgraphs.empty())
        continue;

      subgraphs = policy->shouldFold(subgraphs);
      if (subgraphs.empty())
        continue;

      builder.setInsertionPointToEnd(module->getBody());
      subgraphReplacements.push_back(
          cloneSubgraphs(builder, graph, *module, subgraphs, nameCounter));
    }

    if (module->getBody()->getOperations().empty()) {
      // Remove module to not clutter the output.
      continue;
    }

    ErrorOr<ModelsWithBuffer> modelsOr = lowerSubgraphs(*module);
    if (failed(modelsOr)) {
      mlir::emitError(module->getLoc(),
                      "[ConstantFold] Unable to load models: " +
                          Twine(modelsOr.getError()));
      return signalPassFailure();
    }

    // We have to consolidate rewrites to the same graphop as the rewrites of
    // one subgraph may invalidate ops in another subgraph (due to fold
    // hooks).
    llvm::DenseMap<GraphOp,
                   llvm::SmallVector<std::pair<ModelResult, SubgraphsInfo>>>
        executionResults;
    for (const SubgraphsInfo &subgraphsInfo : subgraphReplacements) {
      const Model *model = findModelByName(modelsOr.get().models,
                                           subgraphsInfo.constantSubgraphName);
      ASSERT_STREAM(model != nullptr,
                    << "Model should exist, make sure names above match.");
      AsyncRT::ErrorDiagnosticOr<ModelResult> valuesToTensorsOr =
          executeModel(ctx, model, subgraphsInfo);
      if (valuesToTensorsOr.isError()) {
        AsyncRT::EncodedDiagnostic err = valuesToTensorsOr.takeError();
        auto decoded = err.decodeLocation();
        auto loc =
            mlir::FileLineColLoc::get(module->getContext(), decoded.filename,
                                      decoded.line, decoded.column);
        mlir::emitError(loc, "[ConstantFold] " + Twine(err.getMessage().get()));
        return signalPassFailure();
      }

      ModelResult valuesToTensors = valuesToTensorsOr.takeValue();

      auto [it, _] = executionResults.try_emplace(
          subgraphsInfo.orig,
          SmallVector<std::pair<ModelResult, SubgraphsInfo>>());

      it->second.push_back({std::move(valuesToTensors), subgraphsInfo});
    }

    // Replace output expressions with constants. Also handles DCE.
    for (auto [graph, results] : executionResults) {
      for (const auto &r : results) {
        const ModelResult &modelResult = r.first;
        const SubgraphsInfo &subgraphsInfo = r.second;
        replaceConstantOps(graph, builder, modelResult, subgraphsInfo);
      }
    }

    // Remove module to not clutter the output.
    module = nullptr;

    // Run canonicalization to uncover more constant folding.
    mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);
    pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
    if (failed(pm.run(getOperation()))) {
      mlir::emitError(getOperation().getLoc(),
                      "[ConstantFold] Failed to run canonicalization pass");
      return signalPassFailure();
    }
  }
}

} // namespace M::MO

std::unique_ptr<mlir::Pass>
M::MO::createConstantFoldPass(std::string_view compiledFrameworkLabel,
                              std::optional<std::string> cacheDir,
                              llvm::ArrayRef<std::string> kernelLibraryPaths) {
  return std::make_unique<ConstantFoldPass>(
      compiledFrameworkLabel, std::move(cacheDir), kernelLibraryPaths);
}
