//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheAttrs.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace Cache;

//===----------------------------------------------------------------------===//
// custom<Base64>
//===----------------------------------------------------------------------===//

/// Print the hash as a base64 string.
static void printBase64(AsmPrinter &printer, StringRef hash) {
  printer.printKeywordOrString(llvm::encodeBase64(hash));
}

/// Parse the base64 hash string and store it into the attr.
static ParseResult parseBase64(AsmParser &parser, std::string &hash) {
  // Decode the base64 bytes. Hashes are often 256 bits (32 bytes) so we can use
  // this as a reasonable default.
  std::vector<char> outBytes;
  outBytes.reserve(32);
  if (parser.parseBase64Bytes(&outBytes))
    return failure();

  // Hashes are almost always <= 64 bytes, so this copy is (while not ideal) not
  // too bad.
  hash.assign(outBytes.begin(), outBytes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// CacheDialect attribute support
//===----------------------------------------------------------------------===//

void CacheDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Cache/CacheDialect/CacheAttrs.cpp.inc"
      >();
}

// This custom parse/print pair is only needed because of #5422
Attribute ConstantHashAttr::parse(AsmParser &parser, Type type) {
  std::vector<char> bytes;
  ShapedType parsedType;
  if (parser.parseLess() || parser.parseBase64Bytes(&bytes))
    return nullptr;

  // Parse an optional additional attribute.
  DictionaryAttr extra;
  if (succeeded(parser.parseOptionalComma()) && parser.parseAttribute(extra))
    return nullptr;

  if (parser.parseColonType(parsedType) || parser.parseGreater())
    return nullptr;

  return ConstantHashAttr::get(parser.getContext(), parsedType,
                               StringRef(&*bytes.begin(), bytes.size()), extra);
}

void ConstantHashAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printBase64(printer, getHash());
  if (getAdditionalData())
    printer << ", " << getAdditionalData();
  printer << " : " << getType() << ">";
}

std::optional<uint64_t> ConstantHashAttr::getOptAlign() const {
  DictionaryAttr dict = getAdditionalData();
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(dict.get("align")))
    return intAttr.getUInt();
  return std::nullopt;
}

ReplaceableAttrIndex ConstantHashAttr::convertToIndex(size_t idx) const {
  return TypedHashIndexAttr::get(getContext(), getType(), idx);
}

ReplaceableAttr
HashIndexAttr::convertFromIndex(ArrayRef<Attribute> attrs) const {
  return llvm::cast<ReplaceableAttr>(attrs[getIndex()]);
}

ReplaceableAttr
TypedHashIndexAttr::convertFromIndex(ArrayRef<mlir::Attribute> attrs) const {
  return llvm::cast<ReplaceableAttr>(attrs[getIndex()]);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Cache/CacheDialect/CacheAttrs.cpp.inc"
