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

#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"

using namespace M;
using namespace MO;

// Most ops are async to keep primitives non-blocking (see
// KeyConceptsInGraphRT.md). The exceptions: transpose (TODO(GEX-2569): make
// async after enforcing high water mark), host layout transform, and the
// collectives, which manage their own cross-device parallelism and must
// complete before the runtime proceeds.
bool M::MO::kernelRequiresSyncLaunch(Operation *op) {
  return isa<TransposeOp>(op) || (isOnHost(op) && isa<LayoutTransformOp>(op)) ||
         isa<AllReduceSumOp>(op) || isa<ReduceScatterSumOp>(op) ||
         isa<Composite::AllReduceAddRMSNormQuantFP8Op>(op) ||
         isa<Composite::DistributedMatmulReduceScatterSumOp>(op) ||
         isa<Composite::DistributedReduceScatterRMSNormOp>(op) ||
         isa<Composite::DistributedAllGatherRMSNormOp>(op) ||
         isa<Composite::DistributedAllGatherRMSNormQuantMXFP8Op>(op) ||
         isa<Composite::DistributedAllGatherRMSNormQuantMXFP6Op>(op) ||
         isa<DistributedBroadcastOp>(op) || isa<DistributedScatterOp>(op) ||
         isa<EPDispatchOp>(op) || isa<EPDispatchFP8Op>(op) ||
         isa<EPDispatchBlockScaledNVOp>(op) ||
         isa<EPDispatchBlockScaledNVDynGlobalScalesOp>(op) ||
         isa<EPDispatchMXFP4Op>(op) || isa<EPCombineOp>(op);
}

template <>
std::optional<SmallVector<APInt>>
M::MO::tryGetConstantBlob(mlir::ElementsAttr attr, DType dType) {
  using namespace M::MO::Detail;
  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr)) {
    SmallVector<APInt> blob = llvm::map_to_vector(
        intElemsAttr.getValues(), [](const llvm::APInt &elem) { return elem; });
    return blob;
  }

  if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    ArrayRef<APInt> data =
        denseAttr.getRawHandle().getBlob()->getDataAs<APInt>();
    return SmallVector<APInt>(data.begin(), data.end());
  }

  return std::nullopt;
}

template <>
std::optional<SmallVector<APFloat>>
M::MO::tryGetConstantBlob(mlir::ElementsAttr attr, DType dType) {
  using namespace M::MO::Detail;
  if (auto floatElemsAttr = dyn_cast<FloatArrayElementsAttr>(attr)) {
    SmallVector<APFloat> blob =
        llvm::map_to_vector(floatElemsAttr.getValues(),
                            [](const llvm::APFloat &elem) { return elem; });
    return blob;
  }

  if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    ArrayRef<APFloat> data =
        denseAttr.getRawHandle().getBlob()->getDataAs<APFloat>();
    return SmallVector<APFloat>(data.begin(), data.end());
  }

  return std::nullopt;
}

bool M::MO::isMutableOperand(Value val) {
  return isa<MO::BufferType, MO::OpaqueType>(val.getType());
}

LogicalResult M::MO::checkReshapeShapesCompatible(
    MOSH::ShapeAttr inShape, MOSH::ShapeAttr outShape,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  std::optional<int64_t> inNumEles =
      MOSH::ShapeAttr::getStaticDim(inShape.getSymbolicNumElements());
  std::optional<int64_t> outNumEles =
      MOSH::ShapeAttr::getStaticDim(outShape.getSymbolicNumElements());
  if (!inNumEles.has_value() || !outNumEles.has_value())
    return success();

  // Allow one -1 dimension for now, which is a fancy parameter.
  if (*inNumEles < 0 || *outNumEles < 0)
    return success();

  if (*inNumEles != *outNumEles)
    return emitError() << "input and output elements do not match: "
                       << *inNumEles << " vs " << *outNumEles;
  return success();
}

ErrorOr<int64_t> M::MO::getReductionAxis(Value input, TypeRange results) {
  auto inputTy = dyn_cast<TensorType>(input.getType());
  if (!inputTy || !inputTy.hasRank())
    return Error("needs a ranked tensor input");
  MOSH::ShapeAttr inShape = inputTy.getShape();

  std::optional<int64_t> axis;
  for (Type type : results) {
    auto resultTy = dyn_cast<TensorType>(type);
    if (!resultTy || !resultTy.hasRank() ||
        resultTy.getRank() != inputTy.getRank())
      return Error("results must be tensors of the input's rank");
    MOSH::ShapeAttr outShape = resultTy.getShape();
    SmallVector<int64_t> changed;
    for (int64_t d = 0, e = inShape.getRank(); d < e; ++d)
      if (outShape[d] != inShape[d])
        changed.push_back(d);
    if (changed.size() != 1 ||
        MOSH::ShapeAttr::getStaticDim(outShape[changed.front()]) != 1)
      return Error("must turn exactly one input dimension into 1");
    if (axis && *axis != changed.front())
      return Error("results reduce different axes");
    axis = changed.front();
  }
  if (!axis)
    return Error("has no results");
  return *axis;
}
