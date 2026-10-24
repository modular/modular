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
// Utilities that don't depend on MO directly but are useful to MO and its
// users. Some may move to `Support` over time.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H

#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <cstdlib>
#include <type_traits>

namespace mlir {
class Operation;
} // namespace mlir

namespace M::MO {

//===----------------------------------------------------------------------===//
// MO Utilities
//===----------------------------------------------------------------------===//

/// True when the kernel built for `op` must launch synchronously (transpose,
/// host layout transform, or a collective). Shared by the MO->MOGG and MO->MAP
/// lowerings so their async policy cannot drift.
bool kernelRequiresSyncLaunch(mlir::Operation *op);

namespace Detail {
template <typename inT, typename outT>
static std::optional<SmallVector<outT>>
denseResourceAttrToVecDirect(DenseResourceElementsAttr attr) {
  // If the resources section got removed in the IR, `blob` will be nullptr.
  if (mlir::AsmResourceBlob *blob = attr.getRawHandle().getBlob()) {
    ArrayRef<inT> data = blob->getDataAs<inT>();
    return SmallVector<outT>(data.begin(), data.end());
  }
  return std::nullopt;
}

/// Reinterpret the `uintStorageType` bits of attr as `semantic`, for FP types
/// with no builtin C++ type.
template <typename uintStorageType, typename outputType>
static SmallVector<outputType>
denseResourceAttrToVecSpecialFloat(DenseResourceElementsAttr attr,
                                   const llvm::fltSemantics &semantic) {
  constexpr int numBitsIn = sizeof(uintStorageType) * CHAR_BIT;
  static_assert(std::is_integral_v<uintStorageType>,
                "Please use uintXX_t to handle special floats.");
  static_assert(std::is_unsigned_v<uintStorageType>,
                "Please use uintXX_t to handle special floats.");
  static_assert(numBitsIn <= 64, "Support only up to 64 bit types");

  ArrayRef<uintStorageType> data =
      attr.getRawHandle().getBlob()->getDataAs<uintStorageType>();
  SmallVector<outputType> out =
      llvm::map_to_vector(data, [&](const uintStorageType &elem) {
        // Asserts if `semantic` doesn't always convert to double losslessly.
        double result =
            APFloat(semantic, APInt(numBitsIn, elem)).convertToDouble();
        return static_cast<outputType>(result);
      });
  return out;
}
} // namespace Detail

/// If `attr` represents a blob of constants, return them in a vector,
/// else return None.
template <typename T>
std::optional<SmallVector<T>> tryGetConstantBlob(mlir::ElementsAttr attr,
                                                 DType dType) {
  using namespace Detail;
  using namespace detail;
  if (auto floatElemsAttr = dyn_cast<FloatArrayElementsAttr>(attr)) {
    SmallVector<double> blob = llvm::map_to_vector(
        floatElemsAttr.getValues(),
        [](const llvm::APFloat &elem) { return elem.convertToDouble(); });
    return SmallVector<T>(blob.begin(), blob.end());
  }

  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr)) {
    SmallVector<int64_t> blob = llvm::map_to_vector(
        intElemsAttr.getValues(),
        [](const llvm::APInt &elem) { return elem.getSExtValue(); });
    return SmallVector<T>(blob.begin(), blob.end());
  }

  if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    return dType.dispatch<std::optional<SmallVector<T>>>()
        .template when<M::DType::si32>([&]() {
          return denseResourceAttrToVecDirect<int32_t, T>(denseAttr);
        })
        .template when<M::DType::si64>([&]() {
          return denseResourceAttrToVecDirect<int64_t, T>(denseAttr);
        })
        .template when<M::DType::bf16>([&]() {
          return denseResourceAttrToVecSpecialFloat<uint16_t, T>(
              denseAttr, APFloat::BFloat());
        })
        .template when<M::DType::f16>([&]() {
          return denseResourceAttrToVecSpecialFloat<uint16_t, T>(
              denseAttr, APFloat::IEEEhalf());
        })
        .template when<M::DType::f32>(
            [&]() { return denseResourceAttrToVecDirect<float, T>(denseAttr); })
        .template when<M::DType::f64>([&]() {
          return denseResourceAttrToVecDirect<double, T>(denseAttr);
        })
        .otherwise([&]() { return std::nullopt; });
  }

  return std::nullopt;
}

template <>
std::optional<SmallVector<APFloat>>
tryGetConstantBlob<APFloat>(mlir::ElementsAttr attr, DType dType);

template <>
std::optional<SmallVector<APInt>>
tryGetConstantBlob<APInt>(mlir::ElementsAttr attr, DType dType);

bool isMutableOperand(Value val);

/// Verifies that `inShape`/`outShape` have the same element count when both
/// are static; shapes with a symbolic dim are assumed valid. Shared by the
/// MO::Static::ReshapeOp and MAP::ReshapeSpecAttr verifiers.
LogicalResult checkReshapeShapesCompatible(
    MOSH::ShapeAttr inShape, MOSH::ShapeAttr outShape,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

/// Returns the axis a reduction of `input` into `results` reduces: the one
/// input dimension that every result keeps as a static 1. Errors on any other
/// shape.
ErrorOr<int64_t> getReductionAxis(Value input, TypeRange results);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOUTILS_H
