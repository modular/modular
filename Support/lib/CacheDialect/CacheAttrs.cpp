//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CacheDialect/CacheAttrs.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace Cache;

/// Print the hash as a base64 string.
static void printBase64(AsmPrinter &printer, StringRef hash) {
  printer.printKeywordOrString(llvm::encodeBase64(hash));
}

/// Parse the base64 hash string and store it into the attr.
static ParseResult parseBase64(AsmParser &parser,
                               FailureOr<std::string> &hash) {
  std::string str;
  if (parser.parseString(&str))
    return failure();

  // Decode the base64 bytes. Hashes are often 256 bits (32 bytes) so we can use
  // this as a reasonable default.
  std::vector<char> outBytes;
  outBytes.reserve(32);
  if (auto err = llvm::decodeBase64(str, outBytes)) {
    return mlir::emitError(
        parser.getEncodedSourceLoc(parser.getCurrentLocation()),
        toString(std::move(err)));
  }

  // Hashes are almost always <= 64 bytes, so this copy is (while not ideal) not
  // too bad.
  hash = std::string(outBytes.begin(), outBytes.end());
  return success();
}

//===----------------------------------------------------------------------===//
// CacheDialect attribute support
//===----------------------------------------------------------------------===//

void CacheDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Support/CacheDialect/CacheAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Support/CacheDialect/CacheAttrs.cpp.inc"
