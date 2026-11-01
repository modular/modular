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
// This file implements the MO dialect types.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MO;

namespace {

ParseResult parseShapeParam(AsmParser &p, MOSH::ShapeAttr &shapeAttr) {
  TypedAttr attr;
  OptionalParseResult res = MOSH::parseShapeAttr(p, attr);
  if (res.has_value() && succeeded(res.value())) {
    shapeAttr = cast<MOSH::ShapeAttr>(attr);
    return success();
  }
  return p.emitError(p.getCurrentLocation())
         << "couldn't parse shape parameter: expected literal shape "
            "reference (i.e. `[...]`)";
}

class TensorLikeTypeParser {
public:
  TensorLikeTypeParser(AsmParser &parser) : parser{parser} {}

  ParseResult parse();

  template <typename TensorLikeType>
  TensorLikeType getType() {
    mlir::MLIRContext *ctx = parser.getContext();

    if (!metadata)
      metadata = mlir::DictionaryAttr::get(ctx);

    return parser.getChecked<TensorLikeType>(startLoc, ctx, shape, dtype,
                                             deviceRef, metadata);
  }

private:
  /// Parse the optional deviceRef and metadata fields.
  ParseResult parseOptionalFields();

  /// Try to parse the current token as the metadata field.
  ParseResult parseMetadata();

  /// Parse the closing '>'.
  ParseResult parseComplete();

private:
  AsmParser &parser;
  mlir::SMLoc startLoc;

  MOSH::ShapeAttr shape;
  DType dtype;
  DeviceRefAttr deviceRef;
  mlir::DictionaryAttr metadata;
};

ParseResult TensorLikeTypeParser::parseComplete() {
  return parser.parseGreater();
}

ParseResult TensorLikeTypeParser::parseMetadata() {
  // Metadata is the last field.
  if (parser.parseAttribute(metadata))
    return failure();

  // A trailing comma means the fields are out of order.
  if (parser.parseOptionalComma())
    return parseComplete();

  parser.emitError(
      parser.getCurrentLocation(),
      "attribute field order should be device followed by metadata");
  return failure();
}

ParseResult TensorLikeTypeParser::parseOptionalFields() {
  if (parser.parseOptionalComma()) {
    // An omitted device defaults to host.
    deviceRef = DeviceRefAttr::getDefault(parser.getContext());
    return parseComplete();
  }

  StringRef deviceLabel;

  // No device keyword, so the field is metadata.
  if (parser.parseOptionalKeyword(&deviceLabel)) {
    // An omitted device defaults to host.
    deviceRef = DeviceRefAttr::getDefault(parser.getContext());
    return parseMetadata();
  }

  DeviceId deviceId;

  if (parser.parseColon() || parser.parseInteger(deviceId))
    return failure();

  deviceRef = DeviceRefAttr::get(parser.getContext(), deviceLabel, deviceId);

  if (parser.parseOptionalComma())
    return parseComplete();

  return parseMetadata();
}

ParseResult TensorLikeTypeParser::parse() {
  startLoc = parser.getCurrentLocation();

  if (parser.parseLess())
    return failure();

  if (parseShapeParam(parser, shape))
    return failure();

  if (parser.parseComma())
    return failure();

  FailureOr<DType> dtypeOr = mlir::FieldParser<DType>::parse(parser);
  if (failed(dtypeOr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "invalid dtype in MO::TensorType");
  }

  dtype = dtypeOr.value();

  return parseOptionalFields();
}

void printTensorLikeType(AsmPrinter &printer, TypedAttr shape, DType dtype,
                         DeviceRefAttr device, mlir::DictionaryAttr metadata) {
  printer << '<';
  MOSH::printShapeTypedAttr(printer, shape);
  printer << ", " << dtype.getAsString();

  if (device && device != DeviceRefAttr::getDefault(device.getContext()))
    printer << ", " << device.getLabel() << ":" << device.getId();
  else if (!device)
    printer << ", INVALID_STATE_NO_DEVICE";

  if (!metadata.empty())
    printer << ", " << metadata;

  printer << '>';
}

} // end namespace

FailureOr<DType> mlir::FieldParser<DType>::parse(AsmParser &parser) {
  StringRef value;
  if (parser.parseKeyword(&value))
    return failure();
  return DType::getFromString(value);
}

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

static std::string asString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << type;
  return str;
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// `MO::TensorType` metadata keys.
static constexpr StringRef kLayoutMetadataName = "layout";
static constexpr StringRef kIOKindMetadataName = "io";
static constexpr StringRef kIsMutableMetadataName = "mut";

/// A stable tensor carries a `UnitAttr` under this key in the metadata
/// `DictionaryAttr` of `MO::TensorType`.
static constexpr StringRef kIsStableMetadataName = "stable";

Type TensorType::parse(AsmParser &parser) {
  TensorLikeTypeParser tensorParser(parser);

  if (tensorParser.parse())
    return {};

  return tensorParser.getType<TensorType>();
}

void TensorType::print(AsmPrinter &printer) const {
  printTensorLikeType(printer, getShape(), getDtype(), getDeviceRef(),
                      getMetadata());
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 MOSH::ShapeAttr shape, DType dtype,
                                 DeviceRefAttr deviceRef,
                                 DictionaryAttr metadata) {
  if (!deviceRef)
    return emitError() << "no device attribute was defined";

  if (!llvm::all_of(shape.getValues(), ShapeAttr::isDim))
    return emitError() << "tensor dimensions must be a getDimType()";

  for (TypedAttr dimAttr : shape.getValues()) {
    if (auto dimInt = ShapeAttr::getStaticDim(dimAttr)) {
      if (*dimInt < 0)
        return emitError()
               << "invalid dimension, expect all dimensions >= 0 got: "
               << *dimInt;
    }
  }

  // Reject the legacy format that kept the device in metadata.
  if (metadata && metadata.contains("device"))
    return emitError()
           << "device attribute no longer resides in the metadata dictionary";

  return success();
}

// TODO: Copied from MGP; align the two.
ErrorOr<DType> M::MO::fromMLIRType(Type mlirType) {
  if (auto unkType = llvm::dyn_cast<NoneType>(mlirType))
    return DType(); // invalid denotes unknown.
  if (auto intType = llvm::dyn_cast<IntegerType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(intType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  if (auto floatType = llvm::dyn_cast<FloatType>(mlirType)) {
    FailureOr<DType> dtype = getEquivalentDType(floatType);
    if (succeeded(dtype) && !(dtype->isInvalid()))
      return *dtype;
  }
  return Error(Twine("unable to represent MLIR Type '") + asString(mlirType) +
               "' as an M::DType type.");
}

ErrorOr<Type> M::MO::toMLIRType(MLIRContext *context, DType dType) {
  if (!dType.isValid())
    return NoneType::get(context); // None denotes unknown.
  if (dType.isInt() || dType.isBool())
    if (Type type = getEquivalentIntegerType(context, dType))
      return type;
  if (dType.isFloat())
    if (Type type = getEquivalentFloatType(context, dType))
      return type;
  return Error(Twine("unable to represent DType '") + dType.getAsString() +
               "' as an MLIR type.");
}

DType M::MO::getDTypeOrInvalid(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy.getDtype();
  else if (auto bufferTy = dyn_cast<BufferType>(ty))
    return bufferTy.getDtype();
  ErrorOr<DType> errOr = fromMLIRType(ty);
  return errOr.isError() ? DType::invalid : *errOr;
}

TypedAttr M::MO::getShape(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy.getShape();
  else if (auto bufferTy = dyn_cast<BufferType>(ty))
    return bufferTy.getShape();

  llvm_unreachable(
      "M::MO::getShape can only take MO::BufferType or MO::TensorType");
}

// Base builder which all others call
TensorType TensorType::get(::mlir::MLIRContext *context,
                           MOSH::ShapeAttr shapeAttr, DType dtype,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  if (!deviceRef)
    deviceRef = DeviceRefAttr::getDefault(context);

  return Base::get(context, std::move(shapeAttr), std::move(dtype),
                   std::move(deviceRef), std::move(metadata));
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ::mlir::MLIRContext *context,
                                  MOSH::ShapeAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(context, shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::get(MOSH::ShapeAttr shapeAttr, DType dtype,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  MLIRContext *ctx = shapeAttr.getContext();
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);
  return get(ctx, shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MOSH::ShapeAttr shapeAttr, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shapeAttr, dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(shapeAttr, dtype, deviceRef, metadata);
}

TensorType TensorType::get(MOSH::ShapeAttr shapeAttr, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());
  return get(shapeAttr, *errOrDType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MOSH::ShapeAttr shapeAttr, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  ErrorOr<DType> dtype = fromMLIRType(elementType);
  if (!dtype) {
    emitError() << dtype.takeError();
    return {};
  }
  if (failed(verify(emitError, shapeAttr, *dtype, deviceRef, metadata)))
    return {};
  return TensorType::get(shapeAttr, *dtype, deviceRef, metadata);
}

TensorType TensorType::get(ArrayRef<int64_t> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto shapeAttr = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, elementType, deviceRef,
                                metadata);
}

TensorType TensorType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto shapeAttr = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, elementType, deviceRef,
                                metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<int64_t> shape,
                                  DType dtype, DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto shapeAttr =
      ShapeAttr::getChecked(emitError, shape, ShapeType::get(context));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, dtype, deviceRef,
                                metadata);
}

TensorType TensorType::get(MLIRContext *context, ArrayRef<TypedAttr> shape,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  auto newShape = ShapeAttr::get(shape, ShapeType::get(context));
  return get(newShape, dtype, deviceRef, metadata);
}

TensorType TensorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context,
                                  ArrayRef<TypedAttr> shape, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto shapeAttr =
      ShapeAttr::getChecked(emitError, shape, ShapeType::get(context));
  if (!shapeAttr)
    return {};
  return TensorType::getChecked(emitError, shapeAttr, dtype, deviceRef,
                                metadata);
}

TensorType TensorType::getScalarType(Type elementType, DeviceRefAttr deviceRef,
                                     DictionaryAttr metadata) {
  return get(ArrayRef<TypedAttr>{}, elementType, deviceRef, metadata);
}

TensorType TensorType::getVectorSI64Type(MLIRContext *ctx, int64_t size,
                                         DeviceRefAttr deviceRef,
                                         DictionaryAttr metadata) {
  auto si64Ty = IntegerType::get(ctx, 64, IntegerType::Signed);
  return get(ArrayRef<TypedAttr>{ShapeAttr::getDim(ctx, size)}, si64Ty,
             deviceRef, metadata);
}

LayoutAttr TensorType::getLayoutAttr() {
  Attribute layoutAttrOrNull = getMetadata().get(kLayoutMetadataName);
  return llvm::dyn_cast_or_null<LayoutAttr>(layoutAttrOrNull);
}

IOKindAttr TensorType::getIOKindAttr() {
  return llvm::dyn_cast_or_null<IOKindAttr>(
      getMetadata().get(kIOKindMetadataName));
}

// TODO: Delete this method.
TensorType TensorType::cloneWith(ArrayRef<int64_t> shape, Type dtype) const {
  auto shapeTy = ShapeType::get(getContext());
  auto shapeAttr = ShapeAttr::get(shape, shapeTy);

  ErrorOr<DType> errOrDType = fromMLIRType(dtype);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as an M::DType: "
                << errOrDType.takeError());
  return TensorType::get(shapeAttr, *errOrDType, getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(ArrayRef<TypedAttr> shapeDims) const {
  auto shapeTy = ShapeType::get(getContext());
  auto shapeAttr = ShapeAttr::get(shapeDims, shapeTy);
  return TensorType::get(shapeAttr, getDtype(), getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(MOSH::ShapeAttr shapeAttr) const {
  return TensorType::get(shapeAttr, getDtype(), getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(DType dtype) const {
  return TensorType::get(getShape(), dtype, getDeviceRef(), getMetadata());
}

TensorType TensorType::cloneWith(LayoutAttr layoutAttr) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kLayoutMetadataName, layoutAttr);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

TensorType TensorType::cloneWithoutLayout() const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.erase(kLayoutMetadataName);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

TensorType TensorType::cloneWith(IOKindAttr ioKindAttr) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kIOKindMetadataName, ioKindAttr);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

TensorType TensorType::cloneWithoutIOKind() const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.erase(kIOKindMetadataName);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

BoolAttr TensorType::getIsMutableAttr() {
  return llvm::dyn_cast_or_null<BoolAttr>(
      getMetadata().get(kIsMutableMetadataName));
}

TensorType TensorType::cloneWithMut(BoolAttr isMutable) const {
  NamedAttrList namedAttrs(getMetadata());
  namedAttrs.set(kIsMutableMetadataName, isMutable);
  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

TensorType TensorType::cloneWithStable(bool isStable) const {
  NamedAttrList namedAttrs(getMetadata());

  if (isStable)
    namedAttrs.set(kIsStableMetadataName, UnitAttr::get(getContext()));
  else
    namedAttrs.erase(kIsStableMetadataName);

  DictionaryAttr newMetadata = namedAttrs.getDictionary(getContext());
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), newMetadata);
}

bool TensorType::isStable() const {
  return getMetadata().contains(kIsStableMetadataName);
}

TensorType TensorType::cloneWith(DictionaryAttr metadataAttr) const {
  return TensorType::get(getShape(), getDtype(), getDeviceRef(), metadataAttr);
}

TensorType TensorType::cloneWith(DeviceRefAttr deviceRef) const {
  return TensorType::get(getShape(), getDtype(), deviceRef, getMetadata());
}

bool TensorType::hasRank() const {
  // verify guarantees a ShapeAttr, which is always ranked. Kept for parity
  // with ShapedTypeInterface.
  return true;
}

size_t TensorType::getRank() const { return getShape().getRank(); }

size_t TensorType::getRankNonZero() const {
  return std::max<size_t>(1, getRank());
}

bool TensorType::hasStaticShape() const {
  return hasRank() && getShape().isStatic();
}

Type TensorType::getElementType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  ASSERT_STREAM(!errOrType.isError(),
                << "element type is not representable as an mlir Type: "
                << errOrType.takeError());
  return *errOrType;
}

SmallPtrSet<StringAttr, 4> TensorType::getUsedParamNames() const {
  SmallPtrSet<StringAttr, 4> res;
  mlir::AttrTypeWalker walker;
  walker.addWalk(
      [&](KGEN::ParamDeclRefAttr attr) { res.insert(attr.getName()); });
  walker.walk(getShape());
  return res;
}

ArrayRef<TypedAttr> TensorType::getDims() const {
  return getShape().getValues();
}

TypedAttr TensorType::getDim(size_t n) const { return getDims()[n]; }

bool TensorType::isParameterizedDim(size_t n) const {
  return isParameterizedDimParam(getDim(n));
}

bool TensorType::isIntegerDim(size_t n) const {
  return isIntegerDimParam(getDim(n));
}

bool TensorType::isParameterizedDimParam(TypedAttr attr) {
  bool isParamOrParamExpr = llvm::isa<KGEN::ParamDeclRefAttr>(attr) ||
                            llvm::isa<KGEN::ParamOperatorAttr>(attr);
  return isParamOrParamExpr && ShapeAttr::isDim(attr);
}

bool TensorType::isIntegerDimParam(TypedAttr attr) {
  return ShapeAttr::getStaticDim(attr).has_value();
}

SmallVector<int64_t> TensorType::getIntDims() const {
  return getShape().getIntDims();
}

int64_t TensorType::getIntDim(size_t n) const {
  return getShape().getIntDim(n);
}

TypedAttr TensorType::getSymbolicNumElements() const {
  return getShape().getSymbolicNumElements();
}

std::optional<int64_t> TensorType::getNumElements() const {
  if (!hasRank())
    return {};

  TypedAttr attr = getSymbolicNumElements();
  auto numElements = ShapeAttr::getStaticDim(attr);
  if (!numElements.has_value())
    return {};

  // We allow one -1 dimension for now, which is a fancy parameter.
  // TODO(15523): maybe remove once -1 is handled more gracefully.
  if (*numElements < 0)
    return {};

  return *numElements;
}

TensorType TensorType::cloneWithNewDim(uint64_t dim, uint64_t size) {
  SmallVector<TypedAttr> shape(getDims());
  shape[dim] = ShapeAttr::getDim(shape.front().getContext(), size);
  auto shapeAttr = ShapeAttr::get(shape, ShapeType::get(getContext()));

  return cloneWith(shapeAttr);
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

LogicalResult ListType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType) {
  // check against things like `!mo.list<f32>`
  auto tensorTy = llvm::dyn_cast<TensorType>(elementType);
  if (!tensorTy) {
    return emitError() << "list can only contain tensor elements, but got "
                       << elementType;
  }

  // check against things like `!mo.list<!mo.tensor<[20, D0], f32>>`
  for (TypedAttr dimAttr : tensorTy.getShape().getValues()) {
    if (llvm::isa<KGEN::ParamDeclRefAttr>(dimAttr))
      return emitError() << "tensor element dimensions cannot have parameter";
  }

  return success();
}

DeviceRefAttr ListType::getDeviceRef() const {
  auto tensorType = llvm::cast<TensorType>(getElementType());
  return tensorType.getDeviceRef();
}

Type ListType::cloneWithNewDevice(DeviceRefAttr device) const {
  auto tensorType = llvm::cast<TensorType>(getElementType());
  return ListType::get(getContext(), tensorType.cloneWithNewDevice(device));
}

//===----------------------------------------------------------------------===//
// BundleType
//===----------------------------------------------------------------------===//

namespace {

/// A bundle whose element tensor types are reconstructible from one
/// shape/dtype/device-range tuple; the compact form shared by the BundleType
/// printer and parser.
struct CompactBundleInfo {
  TypedAttr shapeAttr;
  DType dtype;
  // Non-null.
  DeviceRefAttr deviceRef;
  DictionaryAttr metadata;
  // IDs of elements 0 and N-1.
  DeviceId firstId;
  DeviceId lastId;
};

/// Returns a CompactBundleInfo iff all elements are TensorTypes with identical
/// shape, dtype, device label, and metadata, and contiguous device IDs
/// (`ids[i] == ids[0] + i`).
std::optional<CompactBundleInfo>
computeCompactBundleInfo(ArrayRef<Type> elementTypes) {
  if (elementTypes.empty())
    return std::nullopt;

  auto first = llvm::dyn_cast<TensorType>(elementTypes.front());
  if (!first)
    return std::nullopt;

  DeviceRefAttr firstDevice = first.getDeviceRef();
  if (!firstDevice)
    return std::nullopt;

  DeviceId firstId = firstDevice.getId();
  for (auto [idx, ty] : llvm::enumerate(elementTypes)) {
    auto tensor = llvm::dyn_cast<TensorType>(ty);
    if (!tensor)
      return std::nullopt;
    if (tensor.getShape() != first.getShape())
      return std::nullopt;
    if (tensor.getDtype() != first.getDtype())
      return std::nullopt;
    if (tensor.getMetadata() != first.getMetadata())
      return std::nullopt;
    DeviceRefAttr devRef = tensor.getDeviceRef();
    if (!devRef || devRef.getLabel() != firstDevice.getLabel())
      return std::nullopt;
    if (devRef.getId() != firstId + static_cast<DeviceId>(idx))
      return std::nullopt;
  }

  DeviceId lastId = firstId + static_cast<DeviceId>(elementTypes.size() - 1);
  return CompactBundleInfo{first.getShape(),    first.getDtype(), firstDevice,
                           first.getMetadata(), firstId,          lastId};
}

} // namespace

Type BundleType::parse(AsmParser &parser) {
  if (parser.parseLess() || parser.parseLSquare())
    return {};

  // The verbose form has a tensor type here; the compact form has shape dims,
  // which are not a type.
  Type firstType;
  OptionalParseResult typeRes = parser.parseOptionalType(firstType);
  if (typeRes.has_value()) {
    if (failed(*typeRes))
      return {};

    // Verbose form.
    SmallVector<Type> elementTypes{firstType};
    while (succeeded(parser.parseOptionalComma())) {
      Type ty;
      if (parser.parseType(ty))
        return {};
      elementTypes.push_back(ty);
    }
    if (parser.parseRSquare() || parser.parseGreater())
      return {};
    return BundleType::get(parser.getContext(), elementTypes);
  }

  // Compact form; the outer `[` doubles as the shape's bracket.
  MLIRContext *ctx = parser.getContext();
  auto shapeTy = parser.getBuilder().getType<ShapeType>();
  SmallVector<TypedAttr> shapeDims;
  if (MOSH::parseShapeValue(parser, shapeDims) || parser.parseRSquare())
    return {};
  auto shapeAttr = parser.getChecked<ShapeAttr>(shapeDims, shapeTy);
  if (!shapeAttr)
    return {};

  if (parser.parseComma())
    return {};

  FailureOr<DType> dtypeOr = mlir::FieldParser<DType>::parse(parser);
  if (failed(dtypeOr))
    return {};
  DType dtype = dtypeOr.value();

  // Optional device section: `, <label>:<id>[-<lastId>]`.
  DeviceRefAttr firstDevice;
  DeviceId firstId = 0;
  DeviceId lastId = 0;
  bool hasExplicitDevice = false;
  DictionaryAttr metadata;

  if (succeeded(parser.parseOptionalComma())) {
    StringRef deviceLabel;
    if (succeeded(parser.parseOptionalKeyword(&deviceLabel))) {
      hasExplicitDevice = true;
      if (parser.parseColon() || parser.parseInteger(firstId))
        return {};
      lastId = firstId;

      // AsmParser has no parseMinus, but the lexer tokenizes `-3` as one
      // signed integer whose absolute value is the range end.
      int64_t rangeEnd = 0;
      OptionalParseResult rangeRes = parser.parseOptionalInteger(rangeEnd);
      if (rangeRes.has_value()) {
        if (failed(*rangeRes))
          return {};
        if (rangeEnd >= 0) {
          parser.emitError(parser.getCurrentLocation())
              << "expected '-' before compact bundle device range end";
          return {};
        }
        lastId = static_cast<DeviceId>(-rangeEnd);
        if (lastId < firstId) {
          parser.emitError(parser.getCurrentLocation())
              << "compact bundle device range must be ordered "
              << "(firstId <= lastId)";
          return {};
        }
      }

      firstDevice = DeviceRefAttr::get(ctx, deviceLabel, firstId);

      if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseAttribute(metadata))
          return {};
      }
    } else {
      if (parser.parseAttribute(metadata))
        return {};
    }
  }

  if (parser.parseGreater())
    return {};

  if (!hasExplicitDevice)
    firstDevice = DeviceRefAttr::getDefault(ctx);
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);

  size_t numElements =
      hasExplicitDevice ? static_cast<size_t>(lastId - firstId + 1) : 1;
  SmallVector<Type> elementTypes;
  elementTypes.reserve(numElements);
  for (size_t i = 0; i < numElements; ++i) {
    DeviceRefAttr devRef =
        hasExplicitDevice
            ? DeviceRefAttr::get(ctx, firstDevice.getLabel(), firstId + i)
            : firstDevice;
    auto tensor =
        parser.getChecked<TensorType>(ctx, shapeAttr, dtype, devRef, metadata);
    if (!tensor)
      return {};
    elementTypes.push_back(tensor);
  }
  return BundleType::get(ctx, elementTypes);
}

void BundleType::print(AsmPrinter &printer) const {
  ArrayRef<Type> elementTypes = getElementTypes();

  if (auto info = computeCompactBundleInfo(elementTypes)) {
    printer << '<';
    MOSH::printShapeTypedAttr(printer, info->shapeAttr);
    printer << ", " << info->dtype.getAsString();

    // Elide the device only when the parser would re-derive it: a single
    // element on the default host device.
    auto defaultDev = DeviceRefAttr::getDefault(getContext());
    bool singleDefault =
        elementTypes.size() == 1 && info->deviceRef == defaultDev;
    if (!singleDefault) {
      printer << ", " << info->deviceRef.getLabel() << ':' << info->firstId;
      if (info->firstId != info->lastId)
        printer << '-' << info->lastId;
    }

    if (info->metadata && !info->metadata.empty())
      printer << ", " << info->metadata;

    printer << '>';
    return;
  }

  printer << "<[";
  llvm::interleaveComma(elementTypes, printer,
                        [&](Type ty) { printer.printType(ty); });
  printer << "]>";
}

LogicalResult BundleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<Type> elementTypes) {
  if (elementTypes.empty())
    return emitError() << "bundle must have at least one element";
  bool allTensors = llvm::all_of(
      elementTypes, [](Type t) { return llvm::isa<TensorType>(t); });
  if (!allTensors)
    return emitError() << "bundle elements must all be !mo.tensor types";
  return success();
}

//===----------------------------------------------------------------------===//
// Device list printing / parsing
//===----------------------------------------------------------------------===//

namespace M::MO {

void printDeviceList(AsmPrinter &printer, ArrayRef<DeviceRefAttr> devices) {
  if (devices.empty())
    return;

  // Compact `label:first[-last]` form when labels match and IDs are
  // consecutive.
  StringRef label = devices.front().getLabel();
  DeviceId firstId = devices.front().getId();
  bool isContiguous = true;
  for (auto [i, d] : llvm::enumerate(devices)) {
    if (d.getLabel() != label ||
        d.getId() != firstId + static_cast<DeviceId>(i)) {
      isContiguous = false;
      break;
    }
  }
  if (isContiguous) {
    printer << label << ':' << firstId;
    if (devices.size() > 1)
      printer << '-' << (firstId + devices.size() - 1);
    return;
  }

  llvm::interleaveComma(devices, printer, [&](DeviceRefAttr d) {
    printer << d.getLabel() << ':' << d.getId();
  });
}

ParseResult parseDeviceList(AsmParser &parser,
                            SmallVectorImpl<DeviceRefAttr> &devices) {
  MLIRContext *ctx = parser.getContext();

  // Entries are `label:id` or `label:id-lastId` and may mix, e.g.
  // `gpu:0-3, cpu:0`.
  auto parseOneEntry = [&]() -> ParseResult {
    StringRef label;
    if (parser.parseKeyword(&label))
      return failure();
    DeviceId firstId = 0;
    if (parser.parseColon() || parser.parseInteger(firstId))
      return failure();
    DeviceId lastId = firstId;

    // The lexer tokenizes `-<lastId>` as one signed integer.
    int64_t rangeEnd = 0;
    OptionalParseResult rangeRes = parser.parseOptionalInteger(rangeEnd);
    if (rangeRes.has_value()) {
      if (failed(*rangeRes))
        return failure();
      if (rangeEnd >= 0)
        return parser.emitError(parser.getCurrentLocation(),
                                "expected '-' before compact device range end");
      lastId = static_cast<DeviceId>(-rangeEnd);
      if (lastId < firstId)
        return parser.emitError(
            parser.getCurrentLocation(),
            "compact device range must be ordered (firstId <= lastId)");
    }

    for (DeviceId id = firstId; id <= lastId; ++id)
      devices.push_back(DeviceRefAttr::get(ctx, label, id));
    return success();
  };

  if (parseOneEntry())
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (parseOneEntry())
      return failure();
  }
  return success();
}

} // namespace M::MO

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

Type ScalarType::getDType() const {
  ErrorOr<Type> errOrType = toMLIRType(getContext(), getDtype());
  ASSERT_STREAM(!errOrType.isError(),
                << "scalar type is not representable as an mlir Type: "
                << errOrType.takeError());
  return *errOrType;
}

//===----------------------------------------------------------------------===//
// MOKernelOperandTypeInterface
//===----------------------------------------------------------------------===//

KernelOperandKind TensorType::getKernelOperandKind() const {
  return KernelOperandKind::Tensor;
}
KernelOperandKind BufferType::getKernelOperandKind() const {
  return KernelOperandKind::Buffer;
}
KernelOperandKind ScalarType::getKernelOperandKind() const {
  return KernelOperandKind::Scalar;
}
KernelOperandKind OpaqueType::getKernelOperandKind() const {
  return KernelOperandKind::Opaque;
}
mlir::StringAttr OpaqueType::getKernelOperandOpaqueName() const {
  return getSymbol();
}

//===----------------------------------------------------------------------===//
// MODialect type support
//===----------------------------------------------------------------------===//

void MODialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MODialect/MOTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

BufferType BufferType::cloneWith(DeviceRefAttr deviceRef) const {
  return BufferType::get(getContext(), getShape(), getDtype(), deviceRef,
                         getMetadata());
}

Type BufferType::parse(AsmParser &parser) {
  TensorLikeTypeParser bufferParser(parser);

  if (bufferParser.parse())
    return {};

  return bufferParser.getType<BufferType>();
}

void BufferType::print(AsmPrinter &printer) const {
  printTensorLikeType(printer, getShape(), getDtype(), getDeviceRef(),
                      getMetadata());
}

/// Verifies that the given dimensions represent a valid tensor shape, and
/// that the element type in DType form is also representable as an MLIR type.
LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 MOSH::ShapeAttr shape, DType dtype,
                                 DeviceRefAttr deviceRef,
                                 DictionaryAttr metadata) {
  if (!deviceRef)
    return emitError() << "no device attribute was defined";

  for (TypedAttr dimAttr : shape.getValues()) {
    if (auto dimInt = ShapeAttr::getStaticDim(dimAttr)) {
      if (*dimInt < 0)
        return emitError()
               << "invalid dimension, expect all dimensions >= 0 got: "
               << *dimInt;
    }
  }

  if (!llvm::all_of(shape.getValues(), ShapeAttr::isDim))
    return emitError() << "buffer dimensions must be a getDimType()";

  // Reject the legacy format that kept the device in metadata.
  if (metadata && metadata.contains("device"))
    return emitError()
           << "device attribute no longer resides in the metadata dictionary";

  return success();
}

// Base builder which all others call
BufferType BufferType::get(::mlir::MLIRContext *context, MOSH::ShapeAttr shape,
                           DType dtype, DeviceRefAttr deviceRef,
                           DictionaryAttr metadata) {
  if (!deviceRef)
    deviceRef = DeviceRefAttr::getDefault(context);

  return Base::get(context, std::move(shape), std::move(dtype),
                   std::move(deviceRef), std::move(metadata));
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ::mlir::MLIRContext *context,
                                  MOSH::ShapeAttr shape, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shape, dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(context, shape, dtype, deviceRef, metadata);
}

BufferType BufferType::get(MOSH::ShapeAttr shape, DType dtype,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  MLIRContext *ctx = shape.getContext();
  if (!metadata)
    metadata = DictionaryAttr::get(ctx);
  return get(ctx, shape, dtype, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MOSH::ShapeAttr shape, DType dtype,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  if (failed(verify(emitError, shape, dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(shape, dtype, deviceRef, metadata);
}

BufferType BufferType::get(MOSH::ShapeAttr shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  ErrorOr<DType> errOrDType = fromMLIRType(elementType);
  ASSERT_STREAM(!errOrDType.isError(),
                << "element type is not representable as a DType: "
                << errOrDType.takeError());
  return get(shape, *errOrDType, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MOSH::ShapeAttr shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  ErrorOr<DType> dtype = fromMLIRType(elementType);
  if (!dtype) {
    emitError() << dtype.takeError();
    return {};
  }
  if (failed(verify(emitError, shape, *dtype, deviceRef, metadata)))
    return {};
  return BufferType::get(shape, *dtype, deviceRef, metadata);
}

BufferType BufferType::get(ArrayRef<TypedAttr> shape, Type elementType,
                           DeviceRefAttr deviceRef, DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::get(shape, ShapeType::get(ctx));
  return get(newShape, elementType, deviceRef, metadata);
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<TypedAttr> shape, Type elementType,
                                  DeviceRefAttr deviceRef,
                                  DictionaryAttr metadata) {
  auto ctx = elementType.getContext();
  auto newShape = ShapeAttr::getChecked(emitError, shape, ShapeType::get(ctx));
  if (!newShape)
    return {};
  return BufferType::getChecked(emitError, newShape, elementType, deviceRef,
                                metadata);
}

BufferType BufferType::get(TensorType tensorType) {
  return get(tensorType.getShape(), tensorType.getElementType(),
             tensorType.getDeviceRef(), tensorType.getMetadata());
}

BufferType BufferType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  TensorType tensorType) {
  if (failed(verify(emitError, tensorType.getShape(), tensorType.getDtype(),
                    tensorType.getDeviceRef(), tensorType.getMetadata())))
    return {};
  return BufferType::get(tensorType);
}

TensorType BufferType::toTensorType(BufferType ty) {
  auto shape = ty.getShape();
  auto dtype = ty.getDtype();
  auto deviceRef = ty.getDeviceRef();
  auto metadata = ty.getMetadata();

  return TensorType::get(shape, dtype, deviceRef, metadata);
}

TensorType BufferType::toTensorType() {
  return BufferType::toTensorType(*this);
}

TensorType M::MO::castFromTensorLike(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy;
  return BufferType::toTensorType(cast<MO::BufferType>(ty));
}

TensorType M::MO::dynCastFromTensorLike(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy;
  if (auto bufferTy = dyn_cast<BufferType>(ty))
    return BufferType::toTensorType(bufferTy);
  return nullptr;
}

size_t BufferType::getRank() const { return getShape().getRank(); }

size_t BufferType::getRankNonZero() const {
  size_t rank = getRank();
  return rank == 0 ? rank + 1 : rank;
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

OpaqueType OpaqueType::get(MLIRContext *ctx, StringAttr symbol,
                           DictionaryAttr parameters) {
  if (!parameters)
    parameters = DictionaryAttr::get(ctx);
  return Base::get(ctx, symbol, parameters);
}

OpaqueType OpaqueType::get(StringAttr symbol) {
  auto ctx = symbol.getContext();
  return get(ctx, symbol, DictionaryAttr::get(ctx));
}

OpaqueType OpaqueType::get(StringAttr symbol, DictionaryAttr parameters) {
  return get(symbol.getContext(), symbol, parameters);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MODialect/MOTypes.cpp.inc"
