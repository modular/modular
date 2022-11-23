//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheOps.h"
#include "Cache/CacheDialect/CacheAttrs.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "LLCL/Runtime/Algorithms.h"
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
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Caching regions
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
static AsyncValueRef<LogicalResult>
cacheSingleRegion(Region &r, OpBuilder &builder, Operation *symbol,
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
  WriteableBufferRef bytecode = WriteableBuffer::get();
  // Store the container in bytecode.
  mlir::writeBytecodeToFile(container, *bytecode);

  // Store it.
  auto hashOr = cache.insert(&container.getBodyRegion(), std::move(bytecode));
  auto out = AsyncValueRef<LogicalResult>::allocate(hashOr.getRuntime());
  // Keeping references is safe here because all the memory is owned by the
  // MLIRContext, which is guaranteed to live longer than any of this.
  hashOr.andThen([&, hashOr = hashOr.copy(), out = out.copy()] {
    if (failed(*hashOr)) {
      out.emplace(mlir::emitError(r.getLoc()) << hashOr->getError());
      return;
    }

    SmallVector<RegionHashAttr> hashVec;
    // If we already have some hashes, we have to append to the end of that
    // array.
    auto hashes =
        symbol->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
    if (hashes)
      hashVec = SmallVector<RegionHashAttr>(hashes.begin(), hashes.end());

    hashVec.push_back(builder.getAttr<RegionHashAttr>(
        **hashOr, llvm::makeArrayRef<SymbolRefAttr>(&*uniqueRefs.begin(),
                                                    uniqueRefs.size())));
    symbol->setAttr(getRegionHashAttrName(),
                    builder.getAttr<RegionHashArrayAttr>(hashVec));

    // Finally, erase the container.
    container.erase();

    out.emplace(success());
  });

  return out;
}

AsyncValueRef<LogicalResult>
M::Cache::deflateOp(Operation *symbol, BlobCache<RegionCacheKey> &cache,
                    AsyncValueRef<LogicalResult> chain) {
  auto out = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to not
  // worry about sequencing w.r.t. this operation, they can just pass in the
  // chain.
  chain.andThen([symbol, &cache, out = out.copy(), chain = chain.copy()] {
    if (failed(*chain)) {
      out.emplace(failure());
      return;
    }

    // If the op is already deflated, we're done!
    if (symbol->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName())) {
      out.emplace(success());
      return;
    }

    OpBuilder builder(symbol);
    SmallVector<AnyAsyncValueRef> results;
    results.reserve(symbol->getNumRegions());
    for (Region &r : symbol->getRegions())
      results.push_back(cacheSingleRegion(r, builder, symbol, cache));

    andThenMoving(results,
                  [out = out.copy()](MutableArrayRef<AnyAsyncValueRef> values) {
                    for (auto &v : values) {
                      if (failed(v->get<LogicalResult>()))
                        out.emplace(failure());
                    }
                    out.emplace(success());
                  });
  });

  return out;
}

/// Inflate a single region from `regionHash` and have `r` take its body.
static AsyncValueRef<LogicalResult>
inflateRegion(Region *r, RegionHashAttr regionHash,
              BlobCache<RegionCacheKey> &cache) {
  auto out = AsyncValueRef<LogicalResult>::allocate(cache.getRuntime());

  auto foundOr = cache.find(regionHash.getHash());
  foundOr.andThen([r, regionHash, foundOr = foundOr.copy(), out = out.copy()] {
    if (foundOr->isError())
      out.emplace(mlir::emitError(r->getLoc()) << foundOr->getError());

    // Parse the bytecode for the region.
    BufferRef bytecodeBuf = foundOr->takeValue();
    std::unique_ptr<llvm::MemoryBuffer> bytecode =
        llvm::MemoryBuffer::getMemBuffer(bytecodeBuf->getBuffer(),
                                         /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);

    // Create a dummy block that we can use to inflate container ops.
    Block b;
    if (failed(mlir::readBytecodeFile(
            *bytecode, &b,
            mlir::ParserConfig(r->getContext(), /*verifyAfterParse=*/false))))
      out.emplace(failure());

    // Get the container and take its body.
    ContainerOp container = cast<ContainerOp>(b.front());
    r->takeBody(container.getBodyRegion());

    // Finish up by replacing symbols with their original names.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](SymbolIndexAttr symRef) {
      return regionHash.getSymbols()[symRef.getIndex()];
    });
    r->walk([&](Operation *op) { replacer.replaceElementsIn(op); });
    out.emplace(success());
  });

  return out;
}

AsyncValueRef<LogicalResult>
M::Cache::inflateOp(Operation *cached, BlobCache<RegionCacheKey> &cache,
                    AsyncValueRef<LogicalResult> chain) {
  auto out = AsyncValueRef<LogicalResult>::allocate(cache.getRuntime());

  // Hang the inflation off the input chain.
  chain.andThen([cached, &cache, chain = chain.copy(), out = out.copy()] {
    auto hashes =
        cached->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
    // If the op doesn't have any region hashes on it, we're done.
    if (!hashes)
      return;

    // Fill in the regions on the operation.
    SmallVector<AnyAsyncValueRef> results;
    for (auto [regionHash, region] : llvm::zip(hashes, cached->getRegions()))
      results.push_back(inflateRegion(&region, regionHash, cache));

    // Once all the regions are cached, remove the region hash attr and resolve
    // success/failure.
    andThenMoving(results, [cached, out = out.copy()](
                               MutableArrayRef<AnyAsyncValueRef> values) {
      for (auto &v : values) {
        if (failed(v->get<LogicalResult>())) {
          out.emplace(failure());
          return;
        }
      }

      // Remove the region hash attr.
      cached->removeAttr(getRegionHashAttrName());
      // Done!
      out.emplace(success());
    });
  });

  return out;
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
