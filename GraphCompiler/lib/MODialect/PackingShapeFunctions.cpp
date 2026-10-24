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

#include "GraphCompiler/MODialect/PackingShapeFunctions.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace M::MO;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

#define PARSE_FIELD(result, field, name, parser)                               \
  auto(field) = parser(kgenAttrs, name);                                       \
  if (!(field))                                                                \
    return {};                                                                 \
  (result).field = std::move(*(field));

static std::optional<SmallVector<int64_t>> unpackDims(DictionaryAttr attr,
                                                      StringRef name) {
  if (auto dimsAttr = attr.getAs<DenseIntElementsAttr>(name))
    return llvm::to_vector(dimsAttr.getValues<int64_t>());

  return {};
}

template <typename AttrType>
static std::optional<AttrType> unpack(DictionaryAttr attr, StringRef name) {
  auto result = attr.getAs<AttrType>(name);
  if (!result)
    return {};
  return result;
}

/// High-level target feature queries over a `TargetInfoAttr`.
class TargetQueries {
public:
  TargetQueries(TargetInfoAttr target) : targetInfo{target} {
    assert(target != nullptr);
  }

  bool hasFeature(StringRef feat) const { return targetInfo.hasFeature(feat); }

  bool isArch(StringRef arch) const { return targetInfo.getArch() == arch; }

  bool isOS(StringRef os) const { return targetInfo.getOS() == os; }

  int64_t simdBitWidth() const {
    if (!targetInfo)
      return 0;
    return targetInfo.getSimdBitWidth();
  }

  int64_t simdWidthOf(Type type) const {
    return simdBitWidth() / type.getIntOrFloatBitWidth();
  }

  bool hasNeonInt8Dotprod() const {
    return hasFeature("neon") && hasFeature("dotprod");
  }

  bool hasNeonInt8Matmul() const {
    return hasFeature("neon") && hasFeature("i8mm");
  }

  bool osIsMacos() const { return isOS("darwin") || isOS("macosx"); }

  bool isNeoverseN1() const { return isArch("neoverse-n1"); }

private:
  TargetInfoAttr targetInfo;
};

static bool useVnniFn(const TargetQueries &target, Type aType, Type bType,
                      Type cType) {
  if (target.hasNeonInt8Dotprod() && !target.hasNeonInt8Matmul())
    return ((aType.isSignedInteger(8) && bType.isSignedInteger(8)) ||
            (aType.isUnsignedInteger(8) && bType.isUnsignedInteger(8))) &&
           cType.isSignedInteger(32);

  if (target.hasFeature("avx2"))
    return aType.isUnsignedInteger(8) && bType.isSignedInteger(8) &&
           cType.isSignedInteger(32);

  return false;
}

static bool useI8mmFn(const TargetQueries &target, Type aType, Type bType,
                      Type cType) {
  bool hasNeonInt8Matmul =
      target.hasFeature("neon") && target.hasFeature("i8mm");

  auto neonTypesMatch =
      (aType.isUnsignedInteger(8) && bType.isUnsignedInteger(8)) ||
      (aType.isUnsignedInteger(8) && bType.isSignedInteger(8)) ||
      (aType.isSignedInteger(8) && bType.isSignedInteger(8));

  return hasNeonInt8Matmul && neonTypesMatch;
}

static int64_t ceilDiv(int64_t numerator, int64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static int64_t alignDown(int64_t value, int64_t alignment) {
  return (value / alignment) * alignment;
}

static int64_t alignUp(int64_t numerator, int64_t alignment) {
  return ceilDiv(numerator, alignment) * alignment;
}

//===----------------------------------------------------------------------===//
// pack_conv_filter_shape
//===----------------------------------------------------------------------===//

struct PackConvFilterShapeKgenAttrs {
  SmallVector<int64_t> dilations;
  SmallVector<int64_t> filterShape;
  SmallVector<int64_t> inputShape;
  SmallVector<int64_t> outputShape;
  SmallVector<int64_t> paddings;
  SmallVector<int64_t> strides;

  IntegerAttr numGroups;
};

std::optional<PackConvFilterShapeKgenAttrs>
unpackConvFilterKgenAttrs(DictionaryAttr kgenAttrs) {
  PackConvFilterShapeKgenAttrs result;

  PARSE_FIELD(result, dilations, "dilations", unpackDims);
  PARSE_FIELD(result, filterShape, "filter_shape", unpackDims);
  PARSE_FIELD(result, inputShape, "input_shape", unpackDims);
  PARSE_FIELD(result, outputShape, "output_shape", unpackDims);
  PARSE_FIELD(result, paddings, "paddings", unpackDims);
  PARSE_FIELD(result, strides, "strides", unpackDims);

  PARSE_FIELD(result, numGroups, "num_groups", unpack<IntegerAttr>);

  return result;
}

struct PadConvInfoStatic {
  SmallVector<int64_t> pad;
  SmallVector<int64_t> stride;
  SmallVector<int64_t> dilation;
  int64_t numGroups;

  bool allKnown() const {
    return !ShapedType::isDynamicShape(pad) &&
           !ShapedType::isDynamicShape(stride) &&
           !ShapedType::isDynamicShape(dilation) &&
           !ShapedType::isDynamic(numGroups);
  }
};

/// See `get_micro_kernel_shape` in Kernels/mojo/nn/conv_utils.mojo
static SmallVector<int64_t>
getMicroKernelShape(const TargetQueries &target, int64_t rank, int64_t WO,
                    int64_t F, const PadConvInfoStatic &convAttr,
                    int64_t simdSize) {
  const bool optimizeStaticShapes = !ShapedType::isDynamic(WO) &&
                                    !ShapedType::isDynamic(F) &&
                                    convAttr.allKnown();

  constexpr int64_t numAvx512Registers = 32;
  constexpr int64_t numAvx2Registers = 16;

  auto isNonZero = [](int64_t x) -> bool { return x != 0; };

  if (optimizeStaticShapes) {
    SmallVector<int64_t> padHVal = {convAttr.pad[0], convAttr.pad[2]};
    SmallVector<int64_t> padWVal = {convAttr.pad[1], convAttr.pad[3]};
    const bool hasPadding =
        llvm::any_of(padHVal, isNonZero) || llvm::any_of(padWVal, isNonZero);

    if (target.hasFeature("avx512f")) {
      if (hasPadding) {
        for (int64_t n : llvm::seq<int64_t>(2, 6)) {
          auto m = (numAvx512Registers - 1) / n - 1;
          if (F % (n * simdSize) == 0 && WO <= m)
            return {WO, n};
        }
      }

      return {6, 4};
    }

    if (target.hasFeature("avx2")) {
      if (hasPadding) {
        int64_t minNumResidual = 3 * WO * F;
        int64_t microKernelHeight = -1;
        int64_t microKernelWidth = -1;

        for (int64_t n : llvm::seq<int64_t>(2, 4)) {
          int64_t m = (numAvx2Registers - 1) / n - 1;
          int64_t numResidual = WO * (F % (n * simdSize)) + (WO % m) * F;

          if (numResidual < minNumResidual) {
            microKernelHeight = m;
            microKernelWidth = n;
            minNumResidual = numResidual;
          }
        }

        return {microKernelHeight, microKernelWidth};
      }

      return {4, 3};
    }

    if (target.isNeoverseN1())
      return {8, 2};

    if (target.hasFeature("neon"))
      return {6, 4};

    return {6, 2};
  } else {
    if (target.hasFeature("avx512f"))
      return {6, 4};

    if (target.isNeoverseN1())
      return {8, 2};

    if (target.hasFeature("neon"))
      return {6, 4};

    return {4, 3};
  }
}

static SmallVector<int64_t> reorderPadding(int64_t rank,
                                           ArrayRef<int64_t> pad) {
  if (rank == 1)
    return to_vector(pad);

  if (rank == 2)
    return {pad[0], pad[2], pad[1], pad[3]};

  return {pad[0], pad[2], pad[4], pad[1], pad[3], pad[5]};
}

std::optional<SmallVector<int64_t>>
M::MO::materializePackConvFilterShape(InvokeShapeFuncOp invoke,
                                      TargetInfoAttr targetAttr) {
  std::optional<PackConvFilterShapeKgenAttrs> maybeAttrs =
      unpackConvFilterKgenAttrs(invoke.getKgenParamsAttr());

  if (!maybeAttrs) {
    invoke.emitError("malformed 'pack_conv_filter_shape' shape invocation");
    return {};
  }

  TargetQueries target(targetAttr);

  MO::TensorType filterType = invoke.getInputTensor(0).getType();
  Type filterElementType = filterType.getElementType();
  int64_t filterRank = filterType.getRank();

  int64_t simdSize = target.simdWidthOf(filterElementType);

  // Detect whether the filter is in FCRS/FCQRS layout (F is dim 0) or
  // RSCF/QRSCF layout (F is the last dim).
  MO::LayoutAttr filterLayout = filterType.getLayoutAttr();
  bool isFCRS = filterLayout && (filterLayout.getFormat() == "FCRS" ||
                                 filterLayout.getFormat() == "FCQRS");
  int64_t f = isFCRS ? filterType.getIntDims()[0]
                     : filterType.getIntDims()[filterRank - 1];

  if (ShapedType::isDynamic(f))
    return {};

  PackConvFilterShapeKgenAttrs &attrs = *maybeAttrs;

  int64_t numGroups = attrs.numGroups.getInt();

  if (f % numGroups != 0) {
    invoke.emitError(
        "number of filtes F must be divisible by number of groups");
    return {};
  }

  int64_t fPerGroup = f / numGroups;

  auto convAttr =
      PadConvInfoStatic{/*pad=*/reorderPadding(filterRank - 2, attrs.paddings),
                        /*stride=*/attrs.strides,
                        /*dilation=*/attrs.dilations,
                        /*numGroups=*/numGroups};

  int64_t wo = filterRank == 4 ? attrs.outputShape[2] : ShapedType::kDynamic;

  SmallVector<int64_t> microKernelShape = getMicroKernelShape(
      target, filterRank - 2, wo, attrs.outputShape[filterRank - 1], convAttr,
      simdSize);

  int64_t microKernelWidth = microKernelShape[1];
  int64_t microKernelFSize = microKernelWidth * simdSize;

  // Packed layout is always FRSCf (2D) or FQRSCf (3D):
  //   [0]          = F_packed = numGroups * ceil(fPerGroup / microKernelFSize)
  //   [1..rank-1]  = spatial dims followed by C (i.e., R,S,C or Q,R,S,C)
  //   [rank]       = f = microKernelFSize
  SmallVector<int64_t> packedShape(filterRank + 1, 0);
  packedShape[0] = numGroups * ceilDiv(fPerGroup, microKernelFSize);
  packedShape[filterRank] = microKernelFSize;

  if (isFCRS) {
    // FCRS -> R,S,C and FCQRS -> Q,R,S,C: dims 2..rank-1, then dim 1 (C).
    int64_t pos = 1;
    for (int64_t i = 2; i < filterRank; ++i)
      packedShape[pos++] = filterType.getIntDim(i);
    packedShape[pos] = filterType.getIntDim(1); // C
  } else {
    // RSCF [R,S,C,F] or QRSCF [Q,R,S,C,F]: dims 0..rank-2 are already R,S,C.
    for (int64_t i : llvm::seq<int64_t>(filterRank - 1))
      packedShape[i + 1] = filterType.getIntDim(i);
  }

  if (llvm::any_of(packedShape, [](int64_t x) { return x < 0; }))
    return {};

  return packedShape;
}

//===----------------------------------------------------------------------===//
// pack_conv_transpose_filter_shape
//===----------------------------------------------------------------------===//

static int64_t getDirectConvMicroKernelWidth(const TargetQueries &target) {
  if (target.hasFeature("avx512f"))
    return 4;
  if (target.isNeoverseN1())
    return 2;
  if (target.hasFeature("neon"))
    return 4;
  return 3;
}

/// Computation taken from pack_filter_shape in:
///     Kernels/mojo/nn/conv_transpose.mojo
std::optional<SmallVector<int64_t>>
M::MO::materializePackConvTransposeFilterShape(InvokeShapeFuncOp invoke,
                                               TargetInfoAttr targetAttr) {
  TargetQueries target(targetAttr);

  Value filter = invoke.getInputTensor(0);
  TensorType filterTensor = cast<TensorType>(filter.getType());

  Type filterType = filterTensor.getElementType();
  int64_t filterRank = filterTensor.getRank();
  SmallVector<int64_t> filterDims = filterTensor.getIntDims();

  if (llvm::any_of(filterDims, ShapedType::isDynamic))
    return {};

  int64_t simdSize = target.simdWidthOf(filterType);
  int64_t microKernelWidth = getDirectConvMicroKernelWidth(target);
  int64_t microKernelFSize = microKernelWidth * simdSize;

  int64_t f = filterDims[filterRank - 2];

  constexpr int64_t numGroups = 1;

  int64_t fPerGroup = f / numGroups;

  // FRSCf layout.
  SmallVector<int64_t> packedShape(filterRank + 1, 0);
  packedShape.resize(filterRank + 1, 0);

  packedShape[0] = numGroups * ceilDiv(fPerGroup, microKernelFSize);
  packedShape[filterRank] = microKernelFSize;

  packedShape[filterRank - 1] = filterDims[filterRank - 1];

  for (int64_t i : llvm::seq<int64_t>(filterRank - 2))
    packedShape[i + 1] = filterDims[i];

  return packedShape;
}

//===----------------------------------------------------------------------===//
// pack_matmul_b_shape_func
//===----------------------------------------------------------------------===//

struct PackMatmulShapeKgenAttrs {
  SmallVector<int64_t> aShape;
  SmallVector<int64_t> bShape;
  SmallVector<int64_t> cShape;

  TypeAttr aType;
  TypeAttr bType;
  TypeAttr cType;

  BoolAttr transposeIn0;
};

std::optional<PackMatmulShapeKgenAttrs>
unpackMatmulKgenAttrs(DictionaryAttr kgenAttrs) {
  PackMatmulShapeKgenAttrs result;

  PARSE_FIELD(result, aShape, "a_shape", unpackDims);
  PARSE_FIELD(result, bShape, "b_shape", unpackDims);
  PARSE_FIELD(result, cShape, "c_shape", unpackDims);

  PARSE_FIELD(result, aType, "a_type", unpack<TypeAttr>);
  PARSE_FIELD(result, bType, "b_type", unpack<TypeAttr>);
  PARSE_FIELD(result, cType, "c_type", unpack<TypeAttr>);

  PARSE_FIELD(result, transposeIn0, "transpose_in_0", unpack<BoolAttr>);

  return result;
}

static bool getKernelType(const TargetQueries &target, int64_t m, int64_t n,
                          int64_t k) {
  if (target.hasFeature("avx512f"))
    return m > 0 && m <= 32;

  if (target.hasFeature("neon")) {
    if (target.isNeoverseN1())
      return (k % 4096) == 0;
    else
      return m > 32;
  }

  return false;
}

struct MicroKernelShape {
  int64_t simdRows;
  int64_t simdCols;
};

static MicroKernelShape getMatmulKernelShapeX86(const TargetQueries &target,
                                                bool kernelType) {
  if (target.hasFeature("avx512f")) {
    if (kernelType)
      return MicroKernelShape{8, 3};
    return MicroKernelShape{6, 4};
  }

  return MicroKernelShape{4, 3};
}

static MicroKernelShape getMatmulKernelShapeARM(const TargetQueries &target,
                                                Type aType, Type bType,
                                                Type cType, bool kernelType) {
  if (target.isNeoverseN1()) {
    if (kernelType)
      return MicroKernelShape{4, 4};
    else
      return MicroKernelShape{8, 2};
  } else {
    bool useI8mm = useI8mmFn(target, aType, bType, cType);
    if (useI8mm)
      return MicroKernelShape{4, 6};
    else if (kernelType)
      return MicroKernelShape{6, 4};
    else
      return MicroKernelShape{8, 2};
  }
}

static MicroKernelShape getMatmulKernelShape(const TargetQueries &target,
                                             Type aType, Type bType, Type cType,
                                             bool kernelType) {
  if (target.hasFeature("neon"))
    return getMatmulKernelShapeARM(target, aType, bType, cType, kernelType);
  return getMatmulKernelShapeX86(target, kernelType);
}

struct KernelConfig {
  int64_t kernelRows;
  int64_t kernelCols;
  int64_t simdSize;
};

static KernelConfig getKernelConfig(const TargetQueries &target, Type aType,
                                    Type bType, Type cType, bool kernelType) {
  int64_t simdSize = target.simdWidthOf(cType);

  MicroKernelShape kernelShape =
      getMatmulKernelShape(target, aType, bType, cType, kernelType);

  return KernelConfig{/*kernelRows=*/kernelShape.simdRows,
                      /*kernelCols=*/kernelShape.simdCols * simdSize,
                      /*simdSize=*/simdSize};
}

static int64_t getPackDataSize(const TargetQueries &target, Type type) {
  // The name KB is a known acronym, so avoid the normal variable casing rules.
  constexpr int64_t KB = 1024; // NOLINT

  int64_t sizeOfType = type.getIntOrFloatBitWidth() / CHAR_BIT;

  // TODO: The Mojo kernel caps this at 4KB in debug builds. Mirroring that
  // needs the Mojo build mode; NDEBUG reflects this file's build instead.

  if (target.osIsMacos())
    return (64 * KB) / sizeOfType;

  if (target.hasFeature("neon") || target.hasFeature("avx512f"))
    return (512 * KB) / sizeOfType;

  return (256 * KB) / sizeOfType;
}

static int64_t getMatmulArchFactor(bool useVnni, bool useI8mm) {
  if (useI8mm)
    return 8;
  if (useVnni)
    return 4;
  return 1;
}

static std::pair<int64_t, int64_t>
calculateTileNK(const TargetQueries &target, Type aType, Type bType, Type cType,
                int64_t kernelCols, int64_t n, int64_t k) {
  int64_t packCacheSize = getPackDataSize(target, bType);
  bool useVnni = useVnniFn(target, aType, bType, cType);
  bool useI8mm = useI8mmFn(target, aType, bType, cType);
  int64_t factor = getMatmulArchFactor(useVnni, useI8mm);

  int64_t leastTileN = kernelCols;

  int64_t largestTileK = alignDown(packCacheSize / leastTileN, factor);

  int64_t tileK = std::min(largestTileK, alignUp(k, factor));

  int64_t maxTileNInInnerSize = packCacheSize / tileK / kernelCols;

  int64_t fullDataTileNInInnerSize = ceilDiv(n, kernelCols);

  int64_t tileNInInnerSize =
      std::min(maxTileNInInnerSize, fullDataTileNInInnerSize);

  int64_t tileN = tileNInInnerSize * kernelCols;

  return {tileN, tileK};
}

static bool useAppleAccelerateLib(const TargetQueries &target, Type cType,
                                  Type aType, Type bType) {
  return target.osIsMacos() &&
         (aType.isF32() && bType.isF32() && cType.isF32());
}

std::optional<SmallVector<int64_t>>
M::MO::materializePackMatmulShapeFunc(InvokeShapeFuncOp invoke,
                                      TargetInfoAttr targetAttr) {
  std::optional<PackMatmulShapeKgenAttrs> maybeAttrs =
      unpackMatmulKgenAttrs(invoke.getKgenParamsAttr());

  if (!maybeAttrs) {
    invoke.emitError("malformed 'pack_matmul_b_shape_func' shape invocation");
    return {};
  }

  PackMatmulShapeKgenAttrs &attrs = *maybeAttrs;
  Type aType = attrs.aType.getValue();
  Type bType = attrs.bType.getValue();
  Type cType = attrs.cType.getValue();

  ArrayRef<int64_t> aShape = attrs.aShape;
  ArrayRef<int64_t> bShape = attrs.bShape;

  // Shape materialization cannot return a runtime size.
  if (ShapedType::isDynamicShape(bShape))
    return {};

  // `getKernelType` expects `m == 0` for dynamic M.
  int64_t kernelTypeM = std::max<int64_t>(0, aShape[0]);
  int64_t n = attrs.transposeIn0.getValue() ? bShape[0] : bShape[1];
  int64_t k = attrs.transposeIn0.getValue() ? bShape[1] : bShape[0];

  TargetQueries target(targetAttr);

  bool kernelType = getKernelType(target, kernelTypeM, n, k);
  auto kernelConfig = getKernelConfig(target, aType, bType, cType, kernelType);

  auto tileNK =
      !attrs.transposeIn0.getValue()
          ? calculateTileNK(target, aType, bType, cType,
                            kernelConfig.kernelCols, bShape[1], bShape[0])
          : calculateTileNK(target, aType, bType, cType,
                            kernelConfig.kernelCols, bShape[0], bShape[1]);

  SmallVector<int64_t, 2> output = {bShape[0], bShape[1]};

  if (attrs.transposeIn0.getValue())
    std::swap(output[0], output[1]);

  if (!useAppleAccelerateLib(target, cType, aType, bType)) {
    bool useVnni = useVnniFn(target, aType, bType, cType);
    bool useI8mm = useI8mmFn(target, aType, bType, cType);
    int64_t factor = getMatmulArchFactor(useVnni, useI8mm);

    output[0] = alignUp(output[0], factor);
    output[1] = alignUp(output[1], std::get<0>(tileNK));
  } else {
    std::swap(output[0], output[1]);
  }

  return output;
}
