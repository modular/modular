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
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/SHA256.h"

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

  llvm::SHA256 sha;
  sha.init();

  // If we have a resource, try to avoid copying the data while hashing it.
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr)) {
    DenseResourceElementsHandle resourceHandle = resource.getRawHandle();
    // Casting char to uint8_t is pretty safe - both are byte types.
    if (resourceHandle.getBlob())
      sha.update(resourceHandle.getBlob()->getDataAs<uint8_t>());
  } else {
    // Hash a generic attr.
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << attr;
    sha.update(stringStream.str());
  }

  auto hash = sha.final();
  return {hash.begin(), hash.end()};
}

LLCL::AsyncValueRef<LogicalResult>
Cache::deflateConstant(Operation *constant, BlobCache<DataCacheKey> &cache,
                       LLCL::AsyncValueRef<LogicalResult> chain) {
  auto out = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to not
  // worry about sequencing w.r.t. this operation, they can just pass in the
  // chain.
  chain.andThen([constant, &cache, out = out.copy(), chain = chain.copy()] {
    if (failed(*chain))
      return out.emplace(failure());

    // Use the replacer strategy to replace "large" attributes with the hashed
    // version.
    mlir::AttrTypeReplacer replacer;
    // For now, we only care about DenseResourceElementsAttr because that's how
    // we handle large attributes.
    replacer.addReplacement(
        [&](DenseResourceElementsAttr resourceAttr) -> Attribute {
          mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
          // If the blob isn't there, we shouldn't try caching nothing.
          if (!blob)
            return nullptr;

          BufferRef resourceData = Buffer::get(
              StringRef(blob->getData().data(), blob->getData().size()));
          // Insert the data into the cache.
          auto hashOr = cache.insert(resourceAttr, std::move(resourceData));
          // This is not great - we have to make this sync because MLIR doesn't
          // really have a good way to handle async here.
          await(hashOr);

          // Create a builder so we can create attrs easier.
          OpBuilder builder(constant);

          NamedAttrList additionalAttrs;
          additionalAttrs.set(
              "align", builder.getIntegerAttr(builder.getType<IntegerType>(
                                                  64, IntegerType::Unsigned),
                                              blob->getDataAlignment()));
          additionalAttrs.set(
              "name",
              builder.getStringAttr(resourceAttr.getRawHandle().getKey()));

          auto newAttr = ConstantHashAttr::get(
              resourceAttr.getContext(), resourceAttr.getType(), **hashOr,
              additionalAttrs.getDictionary(resourceAttr.getContext()));
          return newAttr;
        });

    // Do the replacement now.
    replacer.replaceElementsIn(constant);
    out.emplace(success());
  });

  return out;
}

LLCL::AsyncValueRef<LogicalResult>
Cache::inflateConstant(Operation *constant, BlobCache<DataCacheKey> &cache,
                       LLCL::AsyncValueRef<LogicalResult> chain) {
  auto out = AsyncValueRef<LogicalResult>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to not
  // worry about sequencing w.r.t. this operation, they can just pass in the
  // chain.
  chain.andThen([constant, &cache, out = out.copy(), chain = chain.copy()] {
    if (failed(*chain))
      return out.emplace(failure());

    // Use the replacer strategy to replace "large" attributes with the hashed
    // version.
    mlir::AttrTypeReplacer replacer;
    // For now, we only care about DenseResourceElementsAttr because that's how
    // we handle large attributes.
    replacer.addReplacement([&](ConstantHashAttr cacheAttr) -> Attribute {
      // If `out` has already failed, then we have to just stop - this is not
      // recoverable. In theory we could continue to append errors, but that
      // could result in an error explosion so this is the safe option for now.
      // We can always re-evaluate.
      if (out.getPointer()->isReady() && failed(*out))
        return nullptr;

      // Find the data in the cache.
      auto found = cache.find(cacheAttr.getHash());
      await(found);
      if (found->isError()) {
        out.emplace(mlir::emitError(constant->getLoc()) << found->getError());
        return nullptr;
      }
      if (!found->hasValue()) {
        out.emplace(mlir::emitError(constant->getLoc())
                    << "hash '" << llvm::encodeBase64(cacheAttr.getHash())
                    << "' could not be found in the cache");
        return nullptr;
      }

      // Pull out any attributes we might need.
      DictionaryAttr additional = cacheAttr.getAdditionalData();
      IntegerAttr alignAttr = cast<IntegerAttr>(additional.get("align"));
      StringAttr name = cast<StringAttr>(additional.get("name"));

      // The cache owns the data, so in theory we could rely on a cache dialect
      // resource to keep a reference to the data alive as long as the dialect
      // is alive - that would avoid this copy.
      BufferRef buf = found->takeValue();
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopy(
          ArrayRef<char>(buf->getBufferStart(), buf->getBufferSize()),
          alignAttr.getUInt());

      auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
          constant->getContext());

      // Return the new DenseResourceElementsAttr.
      auto newAttr = DenseResourceElementsAttr::get(
          cacheAttr.getType(),
          resourceManager.insert(name.getValue(), std::move(blob)));
      return newAttr;
    });

    // Do the replacement now.
    replacer.replaceElementsIn(constant);
    // If out has not been set to failure (or something else), then set it to
    // success.
    if (!out.getPointer()->isReady())
      out.emplace(success());
  });

  return out;
}

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
      if (isa<SymbolRefAttr, ConstantHashAttr, RegionHashAttr>(attr))
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
  SmallVector<ConstantHashAttr> hashReferences;
  SmallVector<SymbolIndexAttr> refs;

  // Now we walk the symbol and collect all symbol references.
  r.walk([&](Operation *op) {
    op->getAttrDictionary().walkSubAttrs([&](Attribute attr) {
      if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr))
        symbolReferences.push_back(symbolRef);
      if (auto hash = dyn_cast<ConstantHashAttr>(attr))
        hashReferences.push_back(hash);
    });
  });

  // Create a unique set of symbol references while maintaining the order.
  llvm::SetVector<SymbolRefAttr> uniqueSymbolRefs(symbolReferences.begin(),
                                                  symbolReferences.end());
  llvm::SetVector<ConstantHashAttr> uniqueHashRefs(hashReferences.begin(),
                                                   hashReferences.end());

  // Now we'll take the uniqued list of symbols we have and replace attributes
  // with the appropriate SymbolIndexAttr.
  auto replaceSymbolRef = [&](SymbolRefAttr symRef) {
    auto found = llvm::find(uniqueSymbolRefs, symRef);
    assert(found != uniqueSymbolRefs.end());
    return builder.getAttr<SymbolIndexAttr>(
        std::distance(uniqueSymbolRefs.begin(), found));
  };

  auto replaceHashRef = [&](ConstantHashAttr hashRef) {
    auto found = llvm::find(uniqueHashRefs, hashRef);
    assert(found != uniqueHashRefs.end());
    return builder.getAttr<HashIndexAttr>(
        std::distance(uniqueHashRefs.begin(), found));
  };

  // Walk all the ops and replace their symbol refs with symbol indices, and
  // their hash refs with hash indices.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolRefAttr symbolRefAttr) {
    return replaceSymbolRef(symbolRefAttr);
  });
  replacer.addReplacement(
      [&](ConstantHashAttr hashAttr) { return replaceHashRef(hashAttr); });

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
    if (failed(*hashOr))
      return out.emplace(mlir::emitError(r.getLoc()) << hashOr->getError());

    SmallVector<RegionHashAttr> hashVec;
    // If we already have some hashes, we have to append to the end of that
    // array.
    auto hashes =
        symbol->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
    if (hashes)
      hashVec = SmallVector<RegionHashAttr>(hashes.begin(), hashes.end());

    hashVec.push_back(builder.getAttr<RegionHashAttr>(
        **hashOr,
        llvm::makeArrayRef<SymbolRefAttr>(&*uniqueSymbolRefs.begin(),
                                          uniqueSymbolRefs.size()),
        llvm::makeArrayRef<ConstantHashAttr>(&*uniqueHashRefs.begin(),
                                             uniqueHashRefs.size())));
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
                    for (auto &v : values)
                      if (failed(v->get<LogicalResult>()))
                        return out.emplace(failure());

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
      return out.emplace(mlir::emitError(r->getLoc()) << foundOr->getError());
    if (!foundOr->hasValue()) {
      return out.emplace(mlir::emitError(r->getLoc())
                         << "hash '" << llvm::encodeBase64(regionHash.getHash())
                         << "' could not be found in the cache");
    }

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
      return out.emplace(failure());

    // Get the container and take its body.
    ContainerOp container = cast<ContainerOp>(b.front());
    r->takeBody(container.getBodyRegion());

    // Finish up by replacing symbols/hashes with their original attrs.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](SymbolIndexAttr symRef) {
      return regionHash.getSymbols()[symRef.getIndex()];
    });
    replacer.addReplacement([&](HashIndexAttr hashRef) {
      return regionHash.getHashes()[hashRef.getIndex()];
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
      return out.emplace(success());

    // Fill in the regions on the operation.
    SmallVector<AnyAsyncValueRef> results;
    for (auto [regionHash, region] : llvm::zip(hashes, cached->getRegions()))
      results.push_back(inflateRegion(&region, regionHash, cache));

    // Once all the regions are cached, remove the region hash attr and resolve
    // success/failure.
    andThenMoving(results, [cached, out = out.copy()](
                               MutableArrayRef<AnyAsyncValueRef> values) {
      for (auto &v : values)
        if (failed(v->get<LogicalResult>()))
          return out.emplace(failure());

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
