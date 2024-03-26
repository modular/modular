//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheOps.h"
#include "Cache/CacheDialect/CacheAttrs.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Caching data
//===----------------------------------------------------------------------===//

std::string DataCacheKey::hashKey(DataCacheKey::KeyTy key) {
  if (std::holds_alternative<StringRef>(key))
    return std::get<StringRef>(key).str();

  Attribute attr = std::get<Attribute>(key);

  llvm::BLAKE3 hashState;
  hashState.init();

  // If we have a resource, try to avoid copying the data while hashing it.
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr)) {
    DenseResourceElementsHandle resourceHandle = resource.getRawHandle();
    // Casting char to uint8_t is pretty safe - both are byte types.
    if (resourceHandle.getBlob())
      hashState.update(resourceHandle.getBlob()->getDataAs<uint8_t>());
  } else {
    // Hash a generic attr.
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << attr;
    hashState.update(stringStream.str());
  }

  auto hash = hashState.final();
  return {hash.begin(), hash.end()};
}

//===----------------------------------------------------------------------===//
// CacheDialect::registerOps
//===----------------------------------------------------------------------===//

void CacheDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "Cache/CacheDialect/Cache.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ContainerOp
//===----------------------------------------------------------------------===//

void ContainerOp::build(OpBuilder &builder, OperationState &state,
                        Region &body) {
  Region *region = state.addRegion();
  region->takeBody(body);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Cache/CacheDialect/Cache.cpp.inc"
