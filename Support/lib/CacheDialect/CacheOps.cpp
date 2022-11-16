//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CacheDialect/CacheOps.h"
#include "Support/CacheDialect/CacheAttrs.h"
#include "Support/CacheDialect/CacheDialect.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SHA256.h"

using namespace M;
using namespace Cache;

//===----------------------------------------------------------------------===//
// Caching-related functionality
//===----------------------------------------------------------------------===//

std::string RegionCacheKey::hashKey(RegionCacheKey::KeyTy key) {
  if (std::holds_alternative<StringRef>(key))
    return std::get<StringRef>(key).str();

  Region *r = std::get<Region *>(key);
  llvm::SHA256 sha256;
  sha256.init();

  auto hashTypeOrAttr = [&](auto t) {
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << t;
    sha256.update(stringStream.str());
  };

  // Because the location includes the whole path to the file, we have to just
  // hash the filename (no path), line, and column.
  auto hashLocation = [&](Location loc) {
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      std::filesystem::path p(fileLoc.getFilename().str());
      sha256.update(p.filename().string());
      sha256.update(fileLoc.getLine());
      sha256.update(fileLoc.getColumn());
    }
  };

  for (auto arg : r->getArgumentTypes())
    hashTypeOrAttr(arg);

  r->walk([&](Operation *op) {
    // Add the operation's name to the hash.
    sha256.update(op->getName().getStringRef());

    // Hash the op's location
    hashLocation(op->getLoc());

    // Add operand and result types.
    for (Type t : op->getOperandTypes())
      hashTypeOrAttr(t);
    for (Type t : op->getResultTypes())
      hashTypeOrAttr(t);

    // If the op has nested regions then hash those argument types too.
    for (auto &r : op->getRegions())
      for (auto arg : r.getArgumentTypes())
        hashTypeOrAttr(arg);

    // And finally, attribute values.
    op->getAttrDictionary().walkSubAttrs([&](Attribute attr) {
      if (isa<SymbolRefAttr>(attr))
        return;
      hashTypeOrAttr(attr);
    });
  });

  // Finalize the hash.
  std::array<uint8_t, 32> hash = sha256.final();
  return {hash.begin(), hash.end()};
}

/// Walk a region and:
///  - Collect all symbol references from all operations in that region.
///  - Unique the references and assign them indices.
///  - Replace symbol uses with `cache.symbol_ref`
///  - Cache the region.
static LogicalResult
cacheSingleRegion(Region &r, OpBuilder &builder,
                  SmallVectorImpl<RegionHashAttr> &hashes,
                  BlobCache<M::Cache::RegionCacheKey> &cache) {
  SmallVector<SymbolRefAttr> symbolReferences;
  SmallVector<SymbolIndexAttr> refs;

  // Now we walk the symbol and collect all symbol references.
  r.walk([&](Operation *op) {
    op->getAttrDictionary().walkSubAttrs([&](Attribute attr) {
      if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr))
        symbolReferences.push_back(symbolRef);
    });
  });

  // Create a unique set of symbol references while maintaining the order.
  llvm::SetVector<SymbolRefAttr> uniqueRefs(symbolReferences.begin(),
                                            symbolReferences.end());

  // Now we'll take the uniqued list of symbols we have and replace attributes
  // with the appropriate SymbolIndexAttr.
  auto replaceSymbolRef = [&](SymbolRefAttr symRef) {
    auto found = llvm::find(uniqueRefs, symRef);
    assert(found != uniqueRefs.end());
    return builder.getAttr<SymbolIndexAttr>(
        std::distance(uniqueRefs.begin(), found));
  };

  // Walk all the ops and replace their symbol refs with symbol indices.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolRefAttr symbolRefAttr) {
    return replaceSymbolRef(symbolRefAttr);
  });

  r.walk([&](Operation *op) { replacer.replaceElementsIn(op); });

  // Finally, we can store the region. Create an op to hang it off of so we
  // can cache it.
  auto container = builder.create<ContainerOp>(r.getLoc(), r);

  // Create a place to store the bytecode.
  std::string bytecode;
  llvm::raw_string_ostream bytecodeStream(bytecode);
  // Store the container in bytecode.
  mlir::writeBytecodeToFile(container, bytecodeStream);

  // Now create a memory buffer and store it in the cache.
  std::unique_ptr<llvm::MemoryBuffer> mbufOr =
      llvm::MemoryBuffer::getMemBuffer(bytecode);
  if (!mbufOr)
    return mlir::emitError(r.getLoc())
           << "could not get a memory buffer with the bytecode of a region "
              "for caching";

  // Store it.
  auto hashOr = cache.insert(&container.getBodyRegion(), *mbufOr);
  if (failed(hashOr))
    return mlir::emitError(r.getLoc()) << hashOr.getError();

  // Finally, erase the container.
  container.erase();

  hashes.push_back(builder.getAttr<RegionHashAttr>(
      *hashOr, llvm::makeArrayRef<SymbolRefAttr>(&*uniqueRefs.begin(),
                                                 uniqueRefs.size())));
  return success();
}

LogicalResult M::Cache::deflateOp(Operation *symbol,
                                  BlobCache<RegionCacheKey> &cache) {
  OpBuilder builder(symbol);

  SmallVector<RegionHashAttr> hashes;
  for (Region &r : symbol->getRegions())
    if (failed(cacheSingleRegion(r, builder, hashes, cache)))
      return failure();

  symbol->setAttr(getRegionHashAttrName(),
                  builder.getAttr<RegionHashArrayAttr>(hashes));
  return success();
}

/// Inflate a single region from `regionHash` and have `r` take its body.
static LogicalResult inflateRegion(Region *r, RegionHashAttr regionHash,
                                   BlobCache<RegionCacheKey> &cache) {
  auto foundOr = cache.find(regionHash.getHash());
  if (foundOr.isError())
    return mlir::emitError(r->getLoc()) << foundOr.getError();

  // Create a dummy block that we can use to inflate container ops.
  Block b;

  // Parse the bytecode for the region.
  std::unique_ptr<llvm::MemoryBuffer> bytecode = foundOr.takeValue();
  if (failed(mlir::readBytecodeFile(
          *bytecode, &b,
          mlir::ParserConfig(r->getContext(), /*verifyAfterParse=*/false))))
    return failure();

  // Get the container and take its body.
  ContainerOp container = cast<ContainerOp>(b.front());
  r->takeBody(container.getBodyRegion());

  // Finish up by replacing symbols with their original names.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolIndexAttr symRef) {
    return regionHash.getSymbols()[symRef.getIndex()];
  });
  r->walk([&](Operation *op) { replacer.replaceElementsIn(op); });

  return success();
}

LogicalResult M::Cache::inflateOp(Operation *cached,
                                  BlobCache<RegionCacheKey> &cache) {
  auto hashes =
      cached->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
  if (!hashes)
    return success();

  // Fill in the regions on the operation.
  for (auto [regionHash, region] : llvm::zip(hashes, cached->getRegions())) {
    if (failed(inflateRegion(&region, regionHash, cache)))
      return failure();
  }

  // Remove the region hash attr.
  cached->removeAttr(getRegionHashAttrName());

  // Done!
  return success();
}

//===----------------------------------------------------------------------===//
// CacheDialect::registerOps
//===----------------------------------------------------------------------===//

void CacheDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "Support/CacheDialect/Cache.cpp.inc"
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
#include "Support/CacheDialect/Cache.cpp.inc"
