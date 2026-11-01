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
// This file implements the main entrypoint for the MEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MEFToMLIR.h"
#include "MLRT/GraphRT/MEF/MEFReader.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include <cstring>
#include <numeric>
using namespace M;

// MEF requires 64-byte aligned buffers for parsing
static constexpr size_t kMEFBufferAlignment = 64;
static constexpr size_t kMEFBufferAlignmentMask = kMEFBufferAlignment - 1;

//===----------------------------------------------------------------------===//
// MEFToMLIRConverter
//===----------------------------------------------------------------------===//

namespace {
// This class reads a MEF file and converts it to a mlir module.
class MEFToMLIRConverter {
public:
  MEFToMLIRConverter(Location loc)
      : module(ModuleOp::create(loc)),
        resourceManager(DenseResourceElementsHandle::getManagerInterface(
            loc->getContext())) {}

  ModuleOp takeModule() { return module.release(); }

  bool parseHeader(MEFReader &reader);
  bool findSections(MEFReader &reader);
  bool parseLocationPositions();
  bool parseAttributes();
  bool parseTypes();
  bool parsePrimitiveNames();
  bool parseTopLevelOps();
  bool parseRegionBody(Region *region, size_t numResults, MEFReader &bodyReader,
                       MEFReader &attrNameReader, MEFReader &mlirTypesReader);

  bool emitError(Error message) {
    module->emitError() << message;
    return true;
  }

  bool resolveLocation(size_t locOffset, Location &loc) {
    auto it = locations.find(locOffset);
    if (it == locations.end())
      return emitError("invalid location");
    loc = it->second;
    return false;
  }

  bool resolvePrimitiveName(size_t primIndex, Location loc,
                            StringAttr &result) {
    if (primIndex >= primitiveNames.size())
      return mlir::emitError(loc) << "invalid primitive name", true;
    result = primitiveNames[primIndex];
    return false;
  }

  bool resolveAttribute(size_t attrOffset, Location loc, Attribute &result) {
    auto it = attributeAtOffset.find(attrOffset);
    if (it == attributeAtOffset.end())
      return mlir::emitError(loc) << "invalid attribute offset", true;
    result = it->second;
    return false;
  }

  bool resolveMetadataString(size_t offset, Location loc, StringAttr &result) {
    MEFReader stringReader = getSectionReader(MEF::SectionID::kStringsMD);
    return parseNULTerminatedStringAt(offset, stringReader, result);
  }

  bool resolveRegion(size_t invokableIndex, Location loc, Region *&region) {
    if (invokableIndex >= regions.size())
      return mlir::emitError(loc, "invalid invokable index"), true;

    region = regions[invokableIndex];
    return false;
  }

  bool resolveType(size_t typeID, Location loc, Type &result) {
    if (typeID >= types.size())
      return mlir::emitError(loc) << "invalid type ID #" << typeID, true;
    result = types[typeID];
    return false;
  }

  MEFReader getReaderFor(ArrayRef<uint8_t> data) {
    return MEFReader(data,
                     [this](Error message) { emitError(std::move(message)); });
  }

private:
  MLIRContext *getContext() const { return module.get()->getContext(); }
  Location getLoc() const { return module.get()->getLoc(); }

  /// Parse a NUL terminated string into the specified StringAttr, not including
  /// the NUL terminator.
  bool parseNULTerminatedStringAt(size_t position, MEFReader &reader,
                                  StringAttr &result) {
    StringRef resultStr;
    if (reader.parseNULTerminatedStringAt(position, resultStr))
      return true;
    result = StringAttr::get(getContext(), resultStr);
    return false;
  }

  MEFReader getSectionReader(MEF::SectionID ID) {
    ArrayRef<uint8_t> data;
    if (unsigned(ID) < sections.size())
      data = sections[unsigned(ID)];
    return getReaderFor(data);
  }

  bool parseAttribute(size_t offset, const MEFReader &attrSection,
                      Attribute &resultAttr);

  OwningOpRef<ModuleOp> module;

  /// This is a list of sections indexed by the MEF section ID.
  SmallVector<ArrayRef<uint8_t>, 12> sections;

  /// This is list of filenames indexed by occurrence in the kLocationFilenames
  /// section.
  SmallVector<StringRef> locationFilenames;

  /// This is a list of locations identified by their offset in
  /// kLocationPositions.
  DenseMap<size_t, Location> locations;

  /// This captures the parsed attribute at an offset in the ATTRIBUTES section.
  DenseMap<size_t, Attribute> attributeAtOffset;

  /// This is the name for each primitive, indexed by primitive ID.
  SmallVector<StringAttr> primitiveNames;

  /// This is the set of parsed regions indexed by invokable ID.
  SmallVector<Region *> regions;

  /// The is the parsed version of the type list when metadata is enabled.
  std::vector<Type> types;

  /// A manager used for handling resources.
  DenseResourceElementsHandleManager &resourceManager;
};
} // namespace

/// Read the magic number from the start of the MEF file and check it.
bool MEFToMLIRConverter::parseHeader(MEFReader &reader) {
  // Read magic number.
  uint8_t byte1 = 0, byte2 = 0;
  if (reader.parseByte(byte1) || reader.parseByte(byte2))
    return true;

  if (byte1 != MEF::kMagic1 || byte2 != MEF::kMagic2 + MEF::kVersion0)
    return reader.emitError("missing MEF magic number");

  return false;
}

/// Parse the top level section table into the `sections` array, but do not
/// interpret the sections.
bool MEFToMLIRConverter::findSections(MEFReader &reader) {
  while (!reader.empty()) {
    uint8_t sectionID;
    ArrayRef<uint8_t> sectionData;
    if (reader.readSection(sectionID, sectionData))
      return true;

    // Make sure we have space in the section table for this entry and haven't
    // already seen this section.
    if (sectionID >= sections.size())
      sections.resize(sectionID + 1);
    if (sections[sectionID].data())
      return reader.emitError("duplicate section found");

    sections[sectionID] = sectionData;
  }

  return false;
}

bool MEFToMLIRConverter::parseLocationPositions() {
  MEFReader filenameReader =
      getSectionReader(MEF::SectionID::kLocationFilenames);
  MEFReader reader = getSectionReader(MEF::SectionID::kLocationPositions);

  const uint8_t *sectionStart = reader.data();

  // It is common to have many references to the same filename.  Parse and
  // unique each one a single time.
  DenseMap<size_t, StringAttr> filenames;
  while (!reader.empty()) {
    size_t filenameOffset, lineNo, colNo;
    size_t offset = reader.data() - sectionStart;
    if (reader.parseVBR(filenameOffset) || reader.parseVBR(lineNo) ||
        reader.parseVBR(colNo))
      return true;

    auto &filenameEntry = filenames[filenameOffset];
    if (!filenameEntry) {
      if (parseNULTerminatedStringAt(filenameOffset, filenameReader,
                                     filenameEntry))
        return true;
    }
    locations.insert(
        {offset, FileLineColLoc::get(filenameEntry, lineNo, colNo)});
  }
  return false;
}

/// Return the MLIR type for the specified attribute or null if invalid.
static Type getTypeForAttributeKind(MEF::AttributeKind kind, MLIRContext *ctx) {
  switch (kind) {
  default:
    return Type();
  case MEF::AttributeKind::kBool:
    return IntegerType::get(ctx, 1);
  case MEF::AttributeKind::kS8:
  case MEF::AttributeKind::kU8:
  case MEF::AttributeKind::kS16:
  case MEF::AttributeKind::kU16:
  case MEF::AttributeKind::kS32:
  case MEF::AttributeKind::kU32:
  case MEF::AttributeKind::kS64:
  case MEF::AttributeKind::kU64: {
    IntegerType::SignednessSemantics signedness;
    if (MEF::getCategory(kind) == MEF::AttrCategory::Signed)
      signedness = IntegerType::Signed;
    else
      signedness = IntegerType::Unsigned;
    return IntegerType::get(ctx, MEF::getWidthInBits(kind), signedness);
  }
  case MEF::AttributeKind::kF32:
    return mlir::Float32Type::get(ctx);
  case MEF::AttributeKind::kF64:
    return mlir::Float64Type::get(ctx);
  }
}

/// Parse an attribute at the specified offset.
bool MEFToMLIRConverter::parseAttribute(size_t offset,
                                        const MEFReader &attrSection,
                                        Attribute &resultAttr) {
  // Copy the MEFReader (a cheap operation) so we can slice out of its body.
  MEFReader attrReader = getReaderFor(attrSection.getContents());
  if (attrReader.dropBytes(offset))
    return true;

  auto parseListAttr = [&]() -> Attribute {
    size_t length;
    if (attrSection.parseInvertedVBRAt(offset - 1, length))
      return {};

    SmallVector<Attribute> elements;
    for (size_t i = 0; i < length; ++i) {
      size_t elementOffset;
      elements.push_back(Attribute());
      if (attrReader.parseVBR(elementOffset) ||
          parseAttribute(offset - elementOffset, attrSection, elements.back()))
        return {};
    }
    // Produce a DictionaryAttr for the converter.
    return ArrayAttr::get(getContext(), elements);
  };

  auto parseDictionaryAttr = [&]() -> Attribute {
    size_t length;
    if (attrSection.parseInvertedVBRAt(offset - 1, length))
      return {};

    NamedAttrList outAttrList;
    for (size_t i = 0; i < length; ++i) {
      size_t attrOffset, nameOffset;
      Attribute name;
      Attribute value;
      if (attrReader.parseVBR(nameOffset) || attrReader.parseVBR(attrOffset) ||
          parseAttribute(offset - nameOffset, attrSection, name) ||
          parseAttribute(offset - attrOffset, attrSection, value))
        return {};

      // Construct the NamedAttribute
      outAttrList.push_back(NamedAttribute(cast<StringAttr>(name), value));
    }
    // Produce a DictionaryAttr for the converter.
    return outAttrList.getDictionary(getContext());
  };

  std::function<Type()> parseTypeAttr = [&]() -> Type {
    uint8_t typeValue = 0;
    if (attrReader.parseByte(typeValue))
      return {};
    MEF::TypeKind typeKind = MEF::TypeKind(typeValue);

    bool isComplex = typeKind.isComplex();
    typeKind = MEF::TypeKind(typeKind.stripComplex().getValue());

    // Helper for quantized type parsing.
    auto parseStorageAndExpressedType =
        [&]() -> std::optional<std::tuple<Type, Type>> {
      if (auto storageType = parseTypeAttr()) {
        if (auto expressedType = parseTypeAttr())
          return std::make_tuple(storageType, expressedType);
      }
      return std::nullopt;
    };

    // Helper for determining quantized type storage parameters.
    auto getFlagsStorageTypeMinMax = [](Type storageType) {
      auto intType = cast<IntegerType>(storageType);
      auto isSigned = intType.isSigned();
      unsigned storageBitWidth = intType.getWidth();
      int64_t storageTypeMin =
          QuantizedType::getDefaultMinimumForInteger(isSigned, storageBitWidth);
      int64_t storageTypeMax =
          QuantizedType::getDefaultMaximumForInteger(isSigned, storageBitWidth);
      unsigned flags = isSigned ? mlir::quant::QuantizationFlags::Signed : 0;
      return std::make_tuple(flags, storageTypeMin, storageTypeMax);
    };

    Type resultType;
    switch (typeKind.getValue()) {
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, ...)           \
  case MEF::TypeKind::SHORT_NAME:                                              \
    resultType = MLIR_TYPE::get(getContext());                                 \
    break;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

    case MEF::TypeKind::kBool:
      resultType = IntegerType::get(getContext(), 1);
      break;

    case MEF::TypeKind::tensor: {
      auto eltType = parseTypeAttr();
      uint8_t rank;
      if (!eltType || attrReader.parseByte(rank))
        return {};
      SmallVector<int64_t> shape;
      while (rank--) {
        size_t dim;
        if (attrReader.parseVBR(dim))
          return {};
        shape.push_back(int64_t(dim));
      }
      return RankedTensorType::get(shape, eltType);
    }

    case MEF::TypeKind::function: {
      uint8_t numInputs, numResults;
      if (attrReader.parseByte(numInputs) || attrReader.parseByte(numResults))
        return {};

      SmallVector<Type> inputs, results;
      while (numInputs--) {
        inputs.emplace_back() = parseTypeAttr();
        if (!inputs.back())
          return {};
      }
      while (numResults--) {
        results.emplace_back() = parseTypeAttr();
        if (!results.back())
          return {};
      }
      return FunctionType::get(getContext(), inputs, results);
    }

    case MEF::TypeKind::quantized: {
      auto storageAndExpressedTypeOr = parseStorageAndExpressedType();
      if (!storageAndExpressedTypeOr)
        return {};
      auto [storageType, expressedType] = storageAndExpressedTypeOr.value();

      float scale = 0.0;
      ssize_t zeroPoint;

      // The float type is aligned to 4 bytes.
      attrReader.alignTo(4);
      if (attrReader.parseFloat(scale) || attrReader.parseSignedVBR(zeroPoint))
        return {};

      auto [flags, sTypeMin, sTypeMax] = getFlagsStorageTypeMinMax(storageType);
      return UniformQuantizedType::get(flags, storageType, expressedType, scale,
                                       (int64_t)zeroPoint, sTypeMin, sTypeMax);
    }

    case MEF::TypeKind::quantizedPerAxis: {
      auto storageAndExpressedTypeOr = parseStorageAndExpressedType();
      if (!storageAndExpressedTypeOr)
        return {};
      auto [storageType, expressedType] = storageAndExpressedTypeOr.value();

      uint8_t quantizedDim;
      size_t numParams;
      if (attrReader.parseByte(quantizedDim) || attrReader.parseVBR(numParams))
        return {};

      SmallVector<double> scales(numParams);
      for (auto &scale : scales) {
        float value = 0;
        if (attrReader.parseFloat(value))
          return {};
        scale = value;
      }

      SmallVector<int64_t> zeroPoints(numParams);
      for (auto &zeroPoint : zeroPoints) {
        ssize_t zp;
        if (attrReader.parseSignedVBR(zp))
          return {};

        zeroPoint = zp;
      }

      auto [flags, sTypeMin, sTypeMax] = getFlagsStorageTypeMinMax(storageType);
      return UniformQuantizedPerAxisType::get(
          flags, storageType, expressedType, scales, zeroPoints,
          (int32_t)quantizedDim, sTypeMin, sTypeMax);
    }

    case MEF::TypeKind::chain:
      return MEF::ChainType::get(getContext());

    default:
      // Check for signed/unsigned integer.
      if (typeKind.isSInt()) {
        unsigned width = typeKind.getIntegerWidthInBits();
        resultType = IntegerType::get(getContext(), width, IntegerType::Signed);
        break;
      }
      if (typeKind.isUInt()) {
        unsigned width = typeKind.getIntegerWidthInBits();
        resultType =
            IntegerType::get(getContext(), width, IntegerType::Unsigned);
        break;
      }
      // Otherwise unknown type.
      attrReader.emitError("unknown type attribute");
      return {};
    }

    assert(resultType && "Should have found a type");
    if (isComplex)
      resultType = mlir::ComplexType::get(resultType);
    return resultType;
  };

  // Parse a single attribute value from attrReader and return it, or null on
  // an error.
  auto parseAttribute = [&](MEF::AttributeKind attrKind) -> Attribute {
    // Handle strings here. They aren't bitwidth-encoded, so we can just deal
    // with them up front.
    if (attrKind == MEF::AttributeKind::kString ||
        attrKind == MEF::AttributeKind::kSymbolRef) {
      size_t length;
      ArrayRef<uint8_t> result;
      if (attrSection.parseInvertedVBRAt(offset - 1, length) ||
          attrReader.parseBytes(length, result))
        return {};

      StringAttr out = StringAttr::get(
          getContext(), StringRef((const char *)result.data(), result.size()));
      if (attrKind == MEF::AttributeKind::kSymbolRef)
        return FlatSymbolRefAttr::get(out);

      return out;
    }

    if (attrKind == MEF::AttributeKind::kDictionary)
      return parseDictionaryAttr();

    if (attrKind == MEF::AttributeKind::kList)
      return parseListAttr();

    if (attrKind == MEF::AttributeKind::kType) {
      auto type = parseTypeAttr();
      if (!type)
        return {};
      return TypeAttr::get(type);
    }

    uint8_t byte = 0;
    // I1 attrs are a single byte in the MEF format, but are also a separate
    // attribute type, and so are special-cased here.
    if (attrKind == MEF::AttributeKind::kBool) {
      if (attrReader.parseByte(byte))
        return {};
      return BoolAttr::get(getContext(), byte);
    }

    // Calculate the number of bytes the read.
    auto sizeInBits = MEF::getWidthInBits(attrKind);

    // If the bitwidth is not a power of 2 and it's not less than 64, then it's
    // not a valid size. This is very unlikely to happen unless there's a logic
    // flaw elsewhere, i.e. a string that wasn't caught above.
    if (!llvm::isPowerOf2_64(sizeInBits) || sizeInBits > 64) {
      emitError("invalid bitwidth " + Twine(sizeInBits));
      return {};
    }

    unsigned sizeInBytes = sizeInBits / 8;

    // Parse integers of various sizes.
    uint64_t result = 0;
    for (unsigned i = 0; i != sizeInBytes; ++i) {
      if (attrReader.parseByte(byte))
        return {};
      result |= uint64_t(byte) << (i * 8);
    }
    auto resultInt = APInt(sizeInBytes * 8, result);

    if (attrKind == MEF::AttributeKind::kF32)
      return FloatAttr::get(mlir::Float32Type::get(getContext()),
                            APFloat(resultInt.bitsToFloat()));

    if (attrKind == MEF::AttributeKind::kF64)
      return FloatAttr::get(mlir::Float64Type::get(getContext()),
                            APFloat(resultInt.bitsToDouble()));

    return IntegerAttr::get(getTypeForAttributeKind(attrKind, getContext()),
                            resultInt);
  };

  // The attribute kind always precedes the attribute value in a MEF file. In
  // the case of length-delimited types (like string and array) the layout is
  // [length, kind byte, values].
  uint8_t attrKind = 0;
  if (attrSection.parseByteAt(offset - 1, attrKind))
    return true;

  // If this is a scalar attribute value, just parse the single attribute.
  if ((attrKind & (uint8_t)MEF::AttributeKind::mArrayType) == 0) {
    resultAttr = parseAttribute(MEF::AttributeKind(attrKind));
    return resultAttr == Attribute();
  }

  // Otherwise, if this is an array of attributes, decode each of them and then
  // assemble the array.
  size_t length;
  if (attrSection.parseInvertedVBRAt(offset - 1, length))
    return true;
  attrKind &= ~(uint8_t)MEF::AttributeKind::mArrayType;

  auto mefKind = MEF::AttributeKind(attrKind);
  size_t dataElementSize = 1;
  if (mefKind != MEF::AttributeKind::kBool)
    dataElementSize = (MEF::getWidthInBits(mefKind) / 8);

  // Parse the raw data of the attribute.
  ArrayRef<uint8_t> data;
  if (attrReader.parseBytes(length * dataElementSize, data))
    return {};
  auto type =
      ArrayType::get(length, getTypeForAttributeKind(mefKind, getContext()));

  // If the array is large, use a resource for it instead of going through an
  // inline array.
  if (shouldUseOutOfLineAttrStorage(length)) {
    // TODO: In many cases we should be able to use `UnmanagedAsmResourceBlob`
    // here and avoid all allocations. In these cases we need to ensure the
    // incoming MEF doesn't die before the generated MLIR.
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign(data);
    resultAttr = mlir::DenseResourceElementsAttr::get(
        type, resourceManager.insert("mefArray", std::move(blob)));
    return false;
  }

  resultAttr = ArrayElementsAttr::get(data, type);
  return false;
}

bool MEFToMLIRConverter::parseAttributes() {
  MEFReader attrReader = getSectionReader(MEF::SectionID::kAttributes);
  MEFReader attrIndexMDReader =
      getSectionReader(MEF::SectionID::kAttributeIndexMD);

  // Read until we run out of attribute offsets.
  while (!attrIndexMDReader.empty()) {
    size_t offset;
    Attribute resultAttr;
    if (attrIndexMDReader.parseVBR(offset) ||
        parseAttribute(offset, attrReader, resultAttr))
      return true;
    attributeAtOffset[offset] = resultAttr;
  }

  return false;
}

bool MEFToMLIRConverter::parseTypes() {
  MEFReader section = getSectionReader(MEF::SectionID::kMLIRTypeStringsMD);

  // Read until we run out of type strings.
  while (!section.empty()) {
    StringRef typeStr;
    if (section.parseNULTerminatedString(typeStr))
      return true;

    types.push_back(mlir::parseType(typeStr, getContext()));
    if (!types.back()) // parseType already emits an error.
      return true;
  }
  return false;
}

/// Build the "primitiveNames" array by parsing the kPrimitiveNames section,
/// resolving them all to StringAttr's.
bool MEFToMLIRConverter::parsePrimitiveNames() {
  MEFReader reader = getSectionReader(MEF::SectionID::kPrimitiveNames);
  MEFReader stringReader = getSectionReader(MEF::SectionID::kStrings);

  // Read the count of primitive names, and verify it later.
  size_t numPrimitives;
  if (reader.parseVBR(numPrimitives))
    return true;
  primitiveNames.reserve(numPrimitives);

  // Read until we run out of names.
  while (!reader.empty()) {
    // Each primitive name is encoded as an offset into the string table.
    size_t stringOffset;
    StringAttr name;
    if (reader.parseVBR(stringOffset) ||
        parseNULTerminatedStringAt(stringOffset, stringReader, name))
      return true;
    primitiveNames.push_back(name);
  }

  // Check that numPrimitives is right.
  if (numPrimitives != primitiveNames.size())
    return emitError("expected " + Twine(numPrimitives) +
                     " primitives, but found " + Twine(primitiveNames.size()));

  return false;
}

bool MEFToMLIRConverter::parseTopLevelOps() {
  MEFReader reader = getSectionReader(MEF::SectionID::kTopLevelOpIndex);
  MEFReader stringReader = getSectionReader(MEF::SectionID::kStrings);
  MEFReader attrNameReader =
      getSectionReader(MEF::SectionID::kAttributeNamesMD);
  MEFReader mlirTypesReader = getSectionReader(MEF::SectionID::kMLIRTypeMD);
  MEFReader bodyReader = getSectionReader(MEF::SectionID::kTopLevelOpBodies);

  size_t numRegions;
  if (reader.parseVBR(numRegions))
    return true;

  SmallVector<size_t> invokableOffsets;
  invokableOffsets.reserve(numRegions);
  regions.reserve(numRegions);

  SmallVector<Type> inputTypes;
  auto builder = OpBuilder::atBlockBegin(module->getBody());
  Location loc = builder.getUnknownLoc();
  Type unknownType = builder.getType<MEF::UnknownType>();

  // Provides everything we need to create a symbol-having op. Since we want to
  // allow multiple regions per symbol, we have to go through the index section
  // and find all the symbols' bodies and then we can create the ops.
  struct ParsedRegion {
    size_t numArgs, numResults, bodyOffset;
  };

  SmallVector<ParsedRegion> parsedRegions;
  // Read until we run out of invokable descriptors.
  while (!reader.empty()) {
    size_t bodyOffset, numArgs, numResults;
    if (reader.parseVBR(bodyOffset) || reader.parseVBR(numArgs) ||
        reader.parseVBR(numResults))
      return true;

    parsedRegions.push_back(ParsedRegion{numArgs, numResults, bodyOffset});
  }

  // Don't parse the module this way, it already exists! We create these opaque
  // ops here so we can hold onto all the regions we're about to create, then,
  // since the regions are stored in a post-order tree traversal, we will end up
  // resolving the proper parents in the process of parsing parent invokables.
  // This allows us to re-home all the regions and then finally erase the
  // mef.opaque.region ops.
  for (const auto &s : llvm::drop_end(parsedRegions)) {
    OperationState state(loc, "mef.opaque.region");
    // Single region here, add the offset.
    state.addRegion();
    // Create the operation. This object will likely not verify yet, as we don't
    // yet have the function type on the op.
    Operation *theOp = builder.create(state);

    // We parse a single region.
    Region &region = theOp->getRegion(0);
    Block &body = region.emplaceBlock();
    inputTypes.assign(s.numArgs, unknownType);
    body.addArguments(inputTypes, SmallVector<Location, 3>(s.numArgs, loc));
    inputTypes.clear();

    // Add it to the vector now.
    regions.push_back(&region);

    // Parse the invokable body now.
    if (parseRegionBody(&region, s.numResults, bodyReader, attrNameReader,
                        mlirTypesReader))
      return true;

    // If we have a function-like op we have to provide a function type attr.
    if (auto func = dyn_cast<mlir::FunctionOpInterface>(theOp)) {
      // Get the function type from the parsed region. Use this so if we had the
      // types we get those for free.
      auto functionType = builder.getType<FunctionType>(
          func.getFunctionBody().getArgumentTypes(),
          func.getFunctionBody().front().getTerminator()->getOperandTypes());
      // And add it to the invokable.
      func.setFunctionTypeAttr(TypeAttr::get(functionType));
    }
  }

  // The module is special - it's the last one so parse it out.
  if (parseRegionBody(&module->getBodyRegion(), 0, bodyReader, attrNameReader,
                      mlirTypesReader))
    return true;

  // Erase all the temporary placeholder ops.
  for (auto &op : llvm::make_early_inc_range(module->getBodyRegion().front()))
    if (op.getName().getIdentifier() == "mef.opaque.region")
      op.erase();

  // Erase the module's output op, it's not needed.
  module->getBody()->getTerminator()->erase();
  return false;
}

namespace {
class RegionParser {
public:
  RegionParser(Region *region, size_t numResults, MEFReader &bodyReader,
               MEFReader &attrNameReader, MEFReader &mlirTypesReader,
               MEFToMLIRConverter &converter)
      : region(region), numResults(numResults), bodyReader(bodyReader),
        attrNameReader(attrNameReader), mlirTypesReader(mlirTypesReader),
        converter(converter) {}

  /// Entrypoint to parse the invokable body.
  bool parse();

private:
  bool parseRegisterTable();
  bool parsePrimitiveTable(size_t numPrimitives);
  bool parseResultRegs();
  bool parsePrimitiveList();

  Region *region;
  size_t numResults;
  MEFReader &bodyReader;

  MEFReader &attrNameReader;  // A (possibly empty) ATTRIBUTE_NAMES_MD section.
  MEFReader &mlirTypesReader; // A (possibly empty) MLIRTYPE_MD section.
  MEFToMLIRConverter &converter;

  /// This contains the number of operands that must be resolved before the
  /// primitive is executed by the MEF executor.  This is typically equal to the
  /// number of operands in the primitive, but is one for nonstrict ops.
  SmallVector<size_t, 4> primitiveTable;
  /// This contains the register numbers that are output by the invokable.
  SmallVector<size_t> resultRegs;
};
} // namespace

/// Entrypoint to parse the invokable body.
bool RegionParser::parse() {
  size_t numPrimitives;
  return bodyReader.parseVBR(numPrimitives) || parseRegisterTable() ||
         parsePrimitiveTable(numPrimitives) || parseResultRegs() ||
         parsePrimitiveList();
}

/// Parse and ignore the "NumUses" register table.
bool RegionParser::parseRegisterTable() {
  size_t numRegisters;
  if (bodyReader.parseVBR(numRegisters))
    return true;
  while (numRegisters--) {
    size_t numUses;
    if (bodyReader.parseVBR(numUses))
      return true;
  }
  return false;
}

bool RegionParser::parsePrimitiveTable(size_t numPrimitives) {
  while (numPrimitives--) {
    size_t offset, numOperandsBeforeExecuting;
    if (bodyReader.parseVBR(offset) ||
        bodyReader.parseVBR(numOperandsBeforeExecuting))
      return true;
    primitiveTable.push_back(numOperandsBeforeExecuting);
  }

  return false;
}

bool RegionParser::parseResultRegs() {
  for (size_t i = 0; i != numResults; ++i) {
    size_t registerIndex;
    if (bodyReader.parseVBR(registerIndex))
      return true;
    resultRegs.push_back(registerIndex);
  }
  return false;
}

bool RegionParser::parsePrimitiveList() {
  Block &block = region->front();
  auto bodyBuilder = OpBuilder::atBlockBegin(&block);
  Location loc = region->getLoc();
  Type unknownType = MEF::UnknownType::get(bodyBuilder.getContext());
  bool hasMLIRTypes = !mlirTypesReader.empty();

  // This is a table of the SSA values that make up the register table.  MEF
  // doesn't support forward references, so we can just build this top-down.
  SmallVector<Value, 16> registerValues;

  // Read the primitive records.
  for (size_t i = 0, e = primitiveTable.size(); i != e; ++i) {
    if (bodyReader.alignTo(MEF::kPrimitiveEntryAlignment))
      return true;

    bool isOpaqueOp = false;
    uint32_t primitiveID, locOffset;
    uint32_t numOperands, numAttrs, numInvokables, numResults;
    StringAttr primitiveName;
    uint32_t resultRegisterNo = ~0U;
    if (bodyReader.parseInt4(primitiveID) || bodyReader.parseInt4(locOffset) ||
        converter.resolveLocation(locOffset, loc) ||
        converter.resolvePrimitiveName(primitiveID, loc, primitiveName) ||
        bodyReader.parseInt4(numOperands) || bodyReader.parseInt4(numAttrs) ||
        bodyReader.parseInt4(numInvokables) || bodyReader.parseInt4(numResults))
      return true;

    // We read and ignore the # users for each result.
    SmallVector<uint64_t> attrsOffsets;
    SmallVector<uint32_t> numResultUsers, operands, invokablesIndices;
    if (bodyReader.parseInt8Array(numAttrs, attrsOffsets) ||
        bodyReader.parseInt4Array(numOperands, operands) ||
        bodyReader.parseInt4Array(numInvokables, invokablesIndices))
      return true;

    // If the primitive had results, the register number for the first one is
    // emitted.
    if (numResults) {
      if (bodyReader.parseInt4(resultRegisterNo))
        return true;
    }
    if (bodyReader.parseInt4Array(numResults, numResultUsers))
      return true;

    // Resolve all the SSA values for the operands.
    SmallVector<Value> operandValues;
    for (auto operand : operands) {
      if (operand >= registerValues.size())
        return bodyReader.emitError("invalid operand register #");
      operandValues.push_back(registerValues[operand]);
    }

    // The "usedBy" records are only used by the MEF executor, we skip over them
    // without interpreting them.
    auto numResultUserEntries =
        std::accumulate(numResultUsers.begin(), numResultUsers.end(), 0);
    if (bodyReader.dropBytes(numResultUserEntries * sizeof(uint32_t)))
      return true;

    // Resolve all the result types.
    SmallVector<Type> resultTypes;
    if (!hasMLIRTypes) {
      // If we don't have a types section, then just use !mef.unknown type.
      resultTypes.assign(numResults, unknownType);
    } else {
      for (size_t i = 0, e = numResults; i != e; ++i) {
        size_t typeID;
        Type type;
        if (mlirTypesReader.parseVBR(typeID) ||
            converter.resolveType(typeID, loc, type))
          return true;
        resultTypes.push_back(type);
        // If we see a !mef.unknown type, then we assume this operation lacks
        // type information, and therefore needs to be constructed as a
        // mef.opaque operation.
        isOpaqueOp |= isa<MEF::UnknownType>(type);
      }
    }

    // The first primitive is a pseudo op which only defines the argument
    // registers.
    if (i == 0 && region->getNumArguments() != 0) {
      if (numOperands != 0 || numAttrs != 0 || numInvokables != 0 ||
          numResults != region->getNumArguments())
        return bodyReader.emitError("malformed argument pseudo op");

      size_t argNo = 0;
      assert(block.getNumArguments() == numResults);
      for (auto arg : block.getArguments()) {
        assert(resultRegisterNo == registerValues.size() &&
               "result register mismatch");
        ++resultRegisterNo;
        registerValues.push_back(arg);
        arg.setType(resultTypes[argNo++]);
      }
      continue;
    }

    // Resolve attributes.
    SmallVector<Attribute> attrs;
    SmallVector<StringAttr> attrNames;
    for (unsigned attrOffset : attrsOffsets) {
      Attribute attr;
      if (converter.resolveAttribute(attrOffset, loc, attr))
        return true;
      attrs.push_back(attr);
    }

    // If we have attribute names, decode them.
    if (!attrNameReader.empty()) {
      for (size_t attr = 0, e = attrsOffsets.size(); attr != e; ++attr) {
        size_t nameOffset;
        StringAttr name;
        if (attrNameReader.parseVBR(nameOffset) ||
            converter.resolveMetadataString(nameOffset, loc, name))
          return true;
        attrNames.push_back(name);
        // Encoded mef.opaque nodes won't have attribute names.  Reconstruct
        // them as opaque ops.
        isOpaqueOp |= name.getValue().empty();
      }
    }

    // If the invokable has references to symbols or regions, then decode them
    // now. This is safe because we do a post-order traversal of Regions in the
    // tree.
    SmallVector<Region *> regions;
    for (uint32_t invokableIdx : invokablesIndices) {
      Region *invokableRegion;
      if (converter.resolveRegion(invokableIdx, loc, invokableRegion))
        return true;

      regions.push_back(invokableRegion);
    }

    // If this operation has more than one operand but only requires one to be
    // available before executing the primitive, then it is a nonstrict
    // primitive.
    mlir::UnitAttr nonStrict;
    if (operands.size() > 1 && primitiveTable[i] == 1)
      nonStrict = mlir::UnitAttr::get(loc.getContext());

    // Erase any regions that contain themselves so that we don't end up in
    // infinite recursion.
    llvm::erase_if(regions, [&](Region *r) {
      return r == bodyBuilder.getInsertionBlock()->getParent();
    });

    Operation *resultOp;
    IRMapping map;
    if (!hasMLIRTypes || isOpaqueOp) {
      // If we lack type information, create a mef.opaque operation.  We
      // cannot necessarily create the correct operation when we're lacking
      // type info and attribute names because we'll trip the op verifier.
      resultOp =
          MEF::OpaqueOp::create(bodyBuilder, loc, resultTypes, primitiveName,
                                bodyBuilder.getArrayAttr(attrs), nonStrict,
                                operandValues, regions.size());
      // Clone all the region bodies.
      for (auto [index, region] : llvm::enumerate(regions))
        region->cloneInto(&resultOp->getRegion(index), map);
    } else {
      OperationState state(loc, primitiveName, operandValues, resultTypes,
                           /*attrs*/ {});
      for (const auto &[attrName, attr] : llvm::zip(attrNames, attrs))
        state.addAttribute(attrName, attr);

      // Add all the regions.
      for (Region *r : regions)
        r->cloneInto(state.addRegion(), map);

      resultOp = bodyBuilder.create(state);
    }

    // Add all the results to the register table so subsequent ops may use it.
    for (auto result : resultOp->getResults()) {
      assert(resultRegisterNo == registerValues.size() &&
             "result register mismatch");
      ++resultRegisterNo;
      registerValues.push_back(result);
    }
  }

  // Make mef.output that produces the result regs for the invokable.
  SmallVector<Value> operandValues;
  for (auto resultReg : resultRegs) {
    if (resultReg >= registerValues.size())
      return bodyReader.emitError("invalid result register #");
    operandValues.push_back(registerValues[resultReg]);
  }
  MEF::OutputOp::create(bodyBuilder, loc, operandValues);

  return false;
}

bool MEFToMLIRConverter::parseRegionBody(Region *region, size_t numResults,
                                         MEFReader &bodyReader,
                                         MEFReader &attrNameReader,
                                         MEFReader &mlirTypesReader) {

  return RegionParser(region, numResults, bodyReader, attrNameReader,
                      mlirTypesReader, *this)
      .parse();
}

//===----------------------------------------------------------------------===//
// translateMEFToMLIR
//===----------------------------------------------------------------------===//

/// Read the specified a MEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp M::translateMEFToMLIR(ArrayRef<uint8_t> buffer, Location loc) {
  loc.getContext()->loadDialect<MEF::MEFDialect>();

  // Check if the buffer is already 64-byte aligned
  bool isAligned = (uintptr_t(buffer.data()) & kMEFBufferAlignmentMask) == 0;

  // Only create an aligned copy if necessary
  std::unique_ptr<uint8_t[]> alignedStorage;
  ArrayRef<uint8_t> alignedBuffer = buffer;

  if (!isAligned) {
    // Create a 64-byte aligned copy of the buffer
    // Allocate extra space to allow for alignment adjustment
    alignedStorage =
        std::make_unique<uint8_t[]>(buffer.size() + kMEFBufferAlignmentMask);

    // Calculate the aligned start address (64-byte aligned)
    uint8_t *alignedStart = reinterpret_cast<uint8_t *>(
        (uintptr_t(alignedStorage.get()) + kMEFBufferAlignmentMask) &
        ~uintptr_t(kMEFBufferAlignmentMask));

    // Copy the buffer data to the aligned location
    std::memcpy(alignedStart, buffer.data(), buffer.size());

    // Create a new ArrayRef pointing to the aligned copy
    alignedBuffer = ArrayRef<uint8_t>(alignedStart, buffer.size());
  }

  // We currently always get our buffer out of a SourceMgr so we know it is
  // aligned.  This allows alignTo to be efficient and simple.  If we care about
  // supporting non-aligned inputs in the future, we can either copy the input
  // buffer to realign it when necessary or make alignTo more complicated by
  // tracking the amount we're offset from natural alignment.
  assert((uintptr_t(alignedBuffer.data()) & kMEFBufferAlignmentMask) == 0 &&
         "Input buffer pointer must be 64-byte aligned for this logic to work");

  MEFToMLIRConverter converter(loc);
  MEFReader reader = converter.getReaderFor(alignedBuffer);

  // Read/validate the header and section table.
  if (converter.parseHeader(reader) || converter.findSections(reader) ||
      // Parse sections in a specific order, unbound by their order in the file.
      converter.parseLocationPositions() || converter.parseAttributes() ||
      converter.parseTypes() || converter.parsePrimitiveNames() ||
      converter.parseTopLevelOps())
    return {};

  return converter.takeModule();
}
