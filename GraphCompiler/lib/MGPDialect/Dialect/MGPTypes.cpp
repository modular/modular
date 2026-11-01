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

#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Custom parsers
//===----------------------------------------------------------------------===//

/// The distinguished keyword for the unknown dtype.
constexpr const char *kUnknownDType = "unk";

/// Parses a DType, where 'unk' is represented as the none value, and
/// 'invalid' is considered illegal.
static ParseResult parseDType(AsmParser &p, std::optional<DType> &optDType) {
  StringRef keyword;
  llvm::SMLoc dtypeLoc;
  if (p.getCurrentLocation(&dtypeLoc) || p.parseKeyword(&keyword))
    return failure();

  if (keyword == kUnknownDType)
    return success();

  FailureOr<DType> failOrDType = DType::getFromString(keyword);
  if (failed(failOrDType))
    return p.emitError(dtypeLoc) << "unrecognized dtype '" << keyword << "'.";
  if (failOrDType->isInvalid()) {
    return p.emitError(dtypeLoc) << "invalid dtype.";
  }

  optDType.emplace(*failOrDType);
  return success();
}

/// Prints a DType, where the none value is printed as 'unk'.
static void printDType(AsmPrinter &p, std::optional<DType> optDType) {
  p << (optDType ? optDType->getAsString() : kUnknownDType);
}

/// Parses ((<int> | '?') 'x')* (<dtype> | 'unk')
static ParseResult parseDimsAndDType(
    AsmParser &p, SmallVector<std::optional<uint64_t>> &dimsOrEmpty,
    std::optional<DType> &optDType, SmallVector<int64_t> &shapeOrEmpty) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc))
    return failure();

  if (p.parseDimensionList(shapeOrEmpty, /*allowDynamic=*/true,
                           /*withTrailingX=*/true))
    return failure();

  dimsOrEmpty.reserve(shapeOrEmpty.size());
  for (int64_t dim : shapeOrEmpty) {
    if (!isValidRawSignedSize(dim))
      return p.emitError(loc) << "shape dimensions must be non-negative or the "
                                 "distinguished '?' unknown dimension.";
    dimsOrEmpty.push_back(rawSignedSizeToOptSize(dim));
  }

  if (p.getCurrentLocation(&loc) || parseDType(p, optDType))
    return failure();

  return success();
}

/// Prints a list of dimensions followed by a DType, printing an empty
/// dimension as the '?' and an empty DType as 'unk'. The shape is ignored.
static void printDimsAndDType(AsmPrinter &p,
                              ArrayRef<std::optional<uint64_t>> dimsOrEmpty,
                              std::optional<DType> optDType,
                              ArrayRef<int64_t> shapeOrEmpty) {
  for (const auto &dim : dimsOrEmpty)
    p.getStream() << (dim ? std::to_string(*dim) : "?") << "x";
  printDType(p, optDType);
}

ParseResult M::MGP::parseTensorSpec(AsmParser &p, TensorSpecType &tensorSpec) {
  SmallVector<std::optional<uint64_t>> dims;
  std::optional<DType> optDType;
  SmallVector<int64_t> shape;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || parseDimsAndDType(p, dims, optDType, shape))
    return failure();

  // Verify here for a located error; `get` would fail locationless and crash.
  if (failed(TensorSpecType::verify([&]() { return p.emitError(loc); }, dims,
                                    optDType, shape)))
    return failure();

  tensorSpec = TensorSpecType::get(p.getContext(), dims, optDType, shape);
  return success();
}

void M::MGP::printTensorSpec(AsmPrinter &p, TensorSpecType tensorSpec) {
  printDimsAndDType(p, tensorSpec.getDimsOrEmpty(), tensorSpec.getOptDType(),
                    tensorSpec.getShapeOrEmpty());
}

ParseResult M::MGP::parseDeviceTensorTail(AsmParser &p, DeviceRefAttr &device,
                                          bool &mut, bool &stable) {
  device = DeviceRefAttr::getDefault(p.getContext());
  mut = false;
  stable = false;

  // Optional tail: `mut`, `stable`, and/or a device reference, comma-separated,
  // in any order, each at most once.
  if (failed(p.parseOptionalComma()))
    return success();

  while (true) {
    if (succeeded(p.parseOptionalKeyword("mut"))) {
      if (mut)
        return p.emitError(p.getCurrentLocation(), "unexpected repeated `mut`");
      mut = true;
    } else if (succeeded(p.parseOptionalKeyword("stable"))) {
      if (stable)
        return p.emitError(p.getCurrentLocation(),
                           "unexpected repeated `stable`");
      stable = true;
    } else {
      FailureOr<DeviceRefAttr> parsedDevice =
          mlir::FieldParser<DeviceRefAttr>::parse(p);
      if (failed(parsedDevice))
        return p.emitError(p.getCurrentLocation(),
                           "expected `mut`, `stable`, or a device reference");
      if (device != DeviceRefAttr::getDefault(p.getContext()))
        return p.emitError(p.getCurrentLocation(),
                           "unexpected repeated device reference");
      device = *parsedDevice;
    }

    if (failed(p.parseOptionalComma()))
      return success();
  }
}

void M::MGP::printDeviceTensorTail(AsmPrinter &p, DeviceRefAttr device,
                                   bool mut, bool stable) {
  if (device != DeviceRefAttr::getDefault(device.getContext())) {
    p << ", ";
    p.printStrippedAttrOrType(device);
  }
  if (mut)
    p << ", mut";
  if (stable)
    p << ", stable";
}

//===----------------------------------------------------------------------===//
// DeviceGraphType
//===----------------------------------------------------------------------===//

DeviceGraphType DeviceGraphType::get(FunctionType signature) {
  return DeviceGraphType::get(signature.getContext(), signature);
}

DeviceGraphType DeviceGraphType::get(MLIRContext *ctx, TypeRange inputs,
                                     TypeRange results) {
  return DeviceGraphType::get(FunctionType::get(ctx, inputs, results));
}

DeviceGraphType DeviceGraphType::get(MLIRContext *ctx, ValueRange inputs,
                                     ValueRange results) {
  return DeviceGraphType::get(ctx, TypeRange(inputs), TypeRange(results));
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

ErrorOr<std::optional<DType>> DTypeType::fromMlirType(Type mlirType) {
  if (auto unkType = llvm::dyn_cast<NoneType>(mlirType))
    return std::nullopt; // None denotes unknown.
  if (auto intType = llvm::dyn_cast<IntegerType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(intType);
    if (succeeded(dtype))
      return *dtype;
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(floatType);
    if (succeeded(dtype))
      return *dtype;
  }
  // TODO(#4717): Surely there's a better way?
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "unable to represent MLIR type ";
  os << mlirType;
  os << " as a DType.";
  return Error(str);
}

ErrorOr<Type> DTypeType::toMLIRType(MLIRContext *context,
                                    std::optional<DType> optDtype) {
  if (!optDtype)
    return NoneType::get(context); // None denotes unknown.
  if (optDtype->isInt() || optDtype->isBool()) {
    Type type = getEquivalentIntegerType(context, *optDtype);
    if (type != nullptr)
      return type;
  }
  if (optDtype->isFloat()) {
    Type type = getEquivalentFloatType(context, *optDtype);
    if (type != nullptr)
      return type;
  }
  return Error(Twine("unable to represent DType '") + optDtype->getAsString() +
               "' as an MLIR type.");
}

ErrorOrSuccess DTypeType::hasMLIRType(std::optional<DType> optDType) {
  if (!optDType)
    return {}; // None denotes unknown.
  if (optDType->isInt() && hasEquivalentIntegerType(*optDType))
    return {};
  if (optDType->isBool())
    return {};
  if (optDType->isFloat() && hasEquivalentFloatType(*optDType))
    return {};
  return Error(Twine("unable to represent DType '") + optDType->getAsString() +
               "' as an MLIR type.");
}

uint8_t DTypeType::asRawDType(std::optional<DType> optDType) {
  return optDType ? optDType->getValue() : DType().getValue();
}

std::optional<DType> DTypeType::fromRawDType(uint8_t rawDType) {
  DType dtype(rawDType);
  return dtype.isInvalid() ? std::optional<DType>() : dtype;
}

//===----------------------------------------------------------------------===//
// TensorSpecType
//===----------------------------------------------------------------------===//

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult
TensorSpecType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<std::optional<uint64_t>> dimsOrEmpty,
                       std::optional<DType> optDType,
                       ArrayRef<int64_t> shapeOrEmpty) {
  if (dimsOrEmpty.size() > kMaxRank)
    return emitError() << "tensor spec must have a rank between 0 and "
                       << kMaxRank << ".";

  if (dimsOrEmpty.size() != shapeOrEmpty.size())
    return emitError() << "tensor spec dims and shape must have the same rank.";

  for (const std::optional<uint64_t> &dim : dimsOrEmpty) {
    if (!isValidOptSize(dim))
      return emitError() << "tensor spec dimensions must be non-negative or "
                            "the distinguished '?' unknown dimension.";
  }

  for (size_t i = 0; i < dimsOrEmpty.size(); ++i) {
    int64_t rawSignedDim = optSizeToRawSignedSize(dimsOrEmpty[i]);
    if (shapeOrEmpty[i] != rawSignedDim)
      return emitError()
             << "tensor spec dims and shape must have the same dimensions.";
  }

  // Caution: can't inline into if since ErrorOrSuccess is temporary.
  ErrorOrSuccess errOrSuc = DTypeType::hasMLIRType(optDType);
  if (const char *err = errOrSuc.getError())
    return emitError() << err;

  return success();
}

TensorSpecType TensorSpecType::get(MLIRContext *context,
                                   ArrayRef<std::optional<uint64_t>> dims,
                                   std::optional<DType> optDType) {
  SmallVector<int64_t> shape;
  shape.reserve(dims.size());
  for (const std::optional<uint64_t> &dim : dims)
    shape.push_back(optSizeToRawSignedSize(dim));
  return TensorSpecType::get(context, dims, optDType, shape);
}

TensorSpecType TensorSpecType::getChecked(
    function_ref<::mlir::InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<std::optional<uint64_t>> dims, std::optional<DType> optDType) {
  return TensorSpecType::get(context, dims, optDType);
}

TensorSpecType TensorSpecType::get(ArrayRef<int64_t> shape, Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());

  SmallVector<std::optional<uint64_t>> dims;
  dims.reserve(shape.size());
  for (auto dim : shape)
    dims.push_back(rawSignedSizeToOptSize(dim));

  return TensorSpecType::get(elementType.getContext(), dims, *errOrDType,
                             shape);
}

TensorSpecType
TensorSpecType::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                           ArrayRef<int64_t> shape, Type elementType) {
  ErrorOr<std::optional<DType>> errOrDType =
      DTypeType::fromMlirType(elementType);
  if (const auto *err = errOrDType.getError()) {
    emitError() << err;
    return {};
  }
  return TensorSpecType::get(shape, elementType);
}

TensorSpecType TensorSpecType::get(MLIRContext *context,
                                   ArrayRef<uint64_t> rawDims,
                                   uint8_t rawDType) {
  std::optional<DType> dtype = DTypeType::fromRawDType(rawDType);
  SmallVector<std::optional<uint64_t>> dims;
  dims.reserve(rawDims.size());
  for (auto dim : rawDims)
    dims.push_back(rawSizeToOptSize(dim));

  return TensorSpecType::get(context, dims, dtype);
}

TensorSpecType
TensorSpecType::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                           MLIRContext *context, ArrayRef<uint64_t> rawShape,
                           uint8_t rawDType) {
  return TensorSpecType::get(context, rawShape, rawDType);
}

TensorSpecType TensorSpecType::get(IntArrayElementsAttr rawShape,
                                   IntegerAttr rawDTypeOrNull) {
  // A null rawDType means unknown, encoded as the invalid DType.
  uint8_t rawDTypeValue =
      rawDTypeOrNull ? rawDTypeOrNull.getUInt() : DType().getValue();
  return TensorSpecType::get(rawShape.getContext(),
                             rawShape.asArrayRef<uint64_t>(), rawDTypeValue);
}

TensorSpecType
TensorSpecType::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                           IntArrayElementsAttr rawShape,
                           IntegerAttr rawDTypeOrNull) {
  return TensorSpecType::get(rawShape, rawDTypeOrNull);
}

/// ShapedTypeInterface::hasRank. Tensor specs are always ranked.
bool TensorSpecType::hasRank() const { return true; }

/// ShapedTypeInterface::cloneWith. The element type must be representable as
/// a dtype.
ShapedType TensorSpecType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                     Type elementType) const {
  ArrayRef<int64_t> newShape = shape.value_or(getShape());
  Type newElementType = elementType ? elementType : getElementType();
  return TensorSpecType::get(newShape, newElementType);
}

/// ShapedTypeInterface::getElementType. Well defined thanks to verify.
Type TensorSpecType::getElementType() const {
  return *DTypeType::toMLIRType(getContext(), getOptDType());
}

/// ShapedTypeInterface::getShape.
ArrayRef<int64_t> TensorSpecType::getShape() const { return getShapeOrEmpty(); }

ArrayRef<std::optional<uint64_t>> TensorSpecType::getDims() const {
  return getDimsOrEmpty();
}

size_t TensorSpecType::getRank() const { return getDims().size(); }

size_t TensorSpecType::getNumDynamicDims() const {
  return static_cast<size_t>(llvm::count(getDims(), std::nullopt));
}

std::optional<uint64_t> TensorSpecType::operator[](size_t i) const {
  ASSERT_STREAM(i < getDims().size(), << "shape index out of range");
  return getDims()[i];
}

std::optional<uint64_t> TensorSpecType::getNumElements() const {
  uint64_t prod = 1;
  for (const std::optional<uint64_t> &dim : getDims()) {
    if (!dim)
      return {}; // dynamic dimension
    prod *= *dim;
  }
  return prod;
}

std::optional<uint64_t> TensorSpecType::getSizeInBytes() const {
  std::optional<DType> optDType = getOptDType();
  if (!optDType)
    return {}; // dynamic dtype
  std::optional<uint64_t> numElems = getNumElements();
  if (!numElems)
    return {}; // dynamic dimension
  ssize_t bytes = optDType->getSizeInBytes(plainSizeToRuntimeSize(*numElems));
  ASSERT_STREAM(bytes >= 0, << "overflow in getSizeInBytes");
  return static_cast<uint64_t>(bytes);
}

IntArrayElementsAttr TensorSpecType::getRawDims() const {
  Type ui64 = IntegerType::get(getContext(), 64, IntegerType::Unsigned);
  SmallVector<APInt> rawDims;
  for (const std::optional<uint64_t> &dim : getDims()) {
    uint64_t rawDim = optSizeToRawSize(dim);
    rawDims.emplace_back(ui64.getIntOrFloatBitWidth(), rawDim,
                         ui64.isSignedInteger());
  }
  return IntArrayElementsAttr::get(
      ArrayType::get(static_cast<int64_t>(getRank()), ui64), rawDims);
}

IntegerAttr TensorSpecType::getRawDType() const {
  Type ui8 = IntegerType::get(getContext(), 8, IntegerType::Unsigned);
  return IntegerAttr::get(ui8, DTypeType::asRawDType(getOptDType()));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MGPDialect type support
//===----------------------------------------------------------------------===//

void MGPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DeviceBufferType (cont.)
//===----------------------------------------------------------------------===//

/// Implements ShapedType::getShape
ArrayRef<int64_t> DeviceBufferType::getShape() const {
  return ArrayRef<int64_t>();
}
