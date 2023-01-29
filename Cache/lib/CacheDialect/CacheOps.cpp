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
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
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

AsyncValueRef<Chain> Cache::deflateConstant(Operation *constant,
                                            RCRef<DataCache> cache,
                                            AnyAsyncValueRef chain) {
  auto out = AsyncValueRef<Chain>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to not
  // worry about sequencing w.r.t. this operation, they can just pass in the
  // chain.
  std::move(chain).andThenSync([constant, cache = cache.copy(),
                                out = out.copy()](
                                   AsyncValueRef<Chain> &&chain) mutable {
    if (chain.isError())
      return out.setToError(chain.takeDiagnostic());

    // Use the replacer strategy to replace "large" attributes with the
    // hashed version.
    mlir::AttrTypeReplacer replacer;
    // For now, we only care about DenseResourceElementsAttr because that's
    // how we handle large attributes.
    replacer.addReplacement(
        [&](DenseResourceElementsAttr resourceAttr) -> Attribute {
          mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
          // If the blob isn't there, we shouldn't try caching nothing.
          if (!blob)
            return nullptr;

          BufferRef resourceData = Buffer::get(
              StringRef(blob->getData().data(), blob->getData().size()));
          // Insert the data into the cache.
          auto hashOr = cache->insert(resourceAttr, std::move(resourceData));
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
    std::move(out).emplace();
  });

  return out;
}

AsyncValueRef<Chain> Cache::inflateConstant(Operation *constant,
                                            RCRef<DataCache> cache,
                                            AnyAsyncValueRef chain) {
  auto out = AsyncValueRef<Chain>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to not
  // worry about sequencing w.r.t. this operation, they can just pass in the
  // chain.
  std::move(chain).andThenSync([constant, cache = cache.copy(),
                                out = out.copy()](
                                   AsyncValueRef<Chain> &&chain) mutable {
    if (chain.isError())
      return out.setToError(chain.takeDiagnostic());

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
      if (out.getPointer()->isReady() && out.isError())
        return nullptr;

      // Find the data in the cache.
      auto found = cache->find(cacheAttr.getHash());
      await(found);
      if (found->isError()) {
        out.setToError(
            getMLIRDiagnostic(found->takeError(), constant->getLoc()));
        return nullptr;
      }
      if (!found->hasValue()) {
        out.setToError(getMLIRDiagnostic(
            Error("hash '" + llvm::encodeBase64(cacheAttr.getHash()) +
                  "' could not be found in the cache"),
            constant->getLoc()));
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
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
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
      std::move(out).emplace();
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

  for (auto arg : r->getArgumentTypes())
    hashTypeOrAttr(arg);

  r->walk([&](Operation *op) {
    // Add the operation's name to the hash.
    sha256.update(op->getName().getStringRef());

    // Hash the op's location
    hashTypeOrAttr(op->getLoc());

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
    op->getAttrDictionary().walkImmediateSubElements(
        [&](Attribute attr) {
          if (isa<SymbolRefAttr, ConstantHashAttr, RegionHashAttr>(attr))
            return;
          hashTypeOrAttr(attr);
        },
        [](Type) {});
  });

  // Finalize the hash.
  std::array<uint8_t, 32> hash = sha256.final();
  return {hash.begin(), hash.end()};
}

/// Walk a region and:
///  - Collect all symbol/constant references from all operations in that
///  region.
///  - Unique the references and assign them indices.
///  - Replace their uses with indices.
///  - Cache the region.
static AsyncValueRef<Chain> cacheSingleRegion(Region &r, Operation *op,
                                              RCRef<RegionCache> cache) {
  OpBuilder builder(op);
  SmallVector<SymbolRefAttr> symbolReferences;
  SmallVector<ConstantHashAttr> hashReferences;
  SmallVector<SymbolRefAttr> refs;

  // Now we walk the symbol and collect all symbol references.
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](Attribute attr) {
    if (auto symbolRef = dyn_cast<SymbolRefAttr>(attr))
      symbolReferences.push_back(symbolRef);
    else if (auto hash = dyn_cast<ConstantHashAttr>(attr))
      hashReferences.push_back(hash);
  });

  r.walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    for (auto t : op->getOperandTypes())
      walker.walk(t);
    for (auto t : op->getResultTypes())
      walker.walk(t);
  });

  // Create a unique set of symbol references while maintaining the order.
  llvm::SetVector<SymbolRefAttr> uniqueSymbolRefs(symbolReferences.begin(),
                                                  symbolReferences.end());
  llvm::SetVector<ConstantHashAttr> uniqueHashRefs(hashReferences.begin(),
                                                   hashReferences.end());

  // Now we'll take the uniqued list of symbols we have and replace attributes
  // with the appropriate (renamed) SymbolRefAttr.
  auto replaceSymbolRef = [&](SymbolRefAttr symRef) {
    auto found = llvm::find(uniqueSymbolRefs, symRef);
    assert(found != uniqueSymbolRefs.end());
    return builder.getAttr<SymbolRefAttr>(builder.getStringAttr(
        std::to_string(std::distance(uniqueSymbolRefs.begin(), found))));
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
  replacer.addReplacement(replaceSymbolRef);
  replacer.addReplacement(replaceHashRef);

  r.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                               /*replaceTypes=*/true);
  });

  // Finally, we can store the region. Create an op to hang it off of so we
  // can cache it.
  auto container = builder.create<ContainerOp>(r.getLoc(), r);

  // Create a place to store the bytecode.
  WriteableBufferRef bytecode = WriteableBuffer::get();
  // Store the container in bytecode.
  mlir::writeBytecodeToFile(container, *bytecode);

  // Store it.
  auto hashOr = cache->insert(&container.getBodyRegion(), std::move(bytecode));
  auto out = AsyncValueRef<Chain>::allocate(hashOr.getRuntime());
  // Keeping references is safe here because all the memory is owned by the
  // MLIRContext, which is guaranteed to live longer than any of this.
  std::move(hashOr).andThenSync(
      [&r, op, container, uniqueSymbolRefs = std::move(uniqueSymbolRefs),
       uniqueHashRefs = std::move(uniqueHashRefs),
       out = out.copy()](AsyncValueRef<ErrorOr<std::string>> &&hashOr) mutable {
        // Create a new builder because this may run well after the rest of this
        // function.
        OpBuilder builder(op);
        if (failed(*hashOr)) {
          return out.setToError(
              getMLIRDiagnostic(hashOr->takeError(), r.getLoc()));
        }

        SmallVector<RegionHashAttr> hashVec;
        // If we already have some hashes, we have to append to the end of that
        // array.
        auto hashes =
            op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
        if (hashes)
          hashVec = SmallVector<RegionHashAttr>(hashes.begin(), hashes.end());

        hashVec.push_back(builder.getAttr<RegionHashAttr>(
            **hashOr,
            ArrayRef<SymbolRefAttr>(&*uniqueSymbolRefs.begin(),
                                    uniqueSymbolRefs.size()),
            ArrayRef<ConstantHashAttr>(&*uniqueHashRefs.begin(),
                                       uniqueHashRefs.size())));

        auto hashVecAttr = builder.getAttr<RegionHashArrayAttr>(hashVec);
        op->setAttr(getRegionHashAttrName(), hashVecAttr);

        // Finally, erase the container.
        container.erase();

        std::move(out).emplace();
      });

  return out;
}

AsyncValueRef<Chain> M::Cache::deflateOp(Operation *op,
                                         RCRef<RegionCache> cache,
                                         AnyAsyncValueRef chain) {
  auto out = AsyncValueRef<Chain>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to
  // not worry about sequencing w.r.t. this operation, they can just pass in
  // the chain.
  std::move(chain).andThenSync([op, cache = cache.copy(), out = out.copy()](
                                   AnyAsyncValueRef &&chain) mutable {
    if (chain.isError())
      return out.setToError(chain.takeDiagnostic());

    // If the op is already deflated, we're done!
    if (op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName())) {
      std::move(out).emplace();
      return;
    }

    SmallVector<AnyAsyncValueRef> results;
    results.reserve(op->getNumRegions());
    for (Region &r : op->getRegions())
      results.push_back(cacheSingleRegion(r, op, cache.copy()));

    andThenSyncMoving(
        results,
        [out = out.copy()](MutableArrayRef<AnyAsyncValueRef> values) mutable {
          for (auto &v : values)
            if (v.isError())
              return out.setToError(v.takeDiagnostic());

          std::move(out).emplace();
        });
  });

  return out;
}

/// Inflate a single region from `regionHash` and have `r` take its body.
static AsyncValueRef<Chain> inflateRegion(Region *r, RegionHashAttr regionHash,
                                          RCRef<RegionCache> cache) {
  auto out = AsyncValueRef<Chain>::allocate(cache->getRuntime());

  auto foundOr = cache->find(regionHash.getHash());
  std::move(foundOr).andThenSync(
      [r, regionHash,
       out = out.copy()](AsyncValueRef<CacheFindResult> &&foundOr) mutable {
        if (foundOr->isError()) {
          return out.setToError(
              getMLIRDiagnostic(foundOr->takeError(), r->getLoc()));
        }
        if (!foundOr->hasValue()) {
          return out.setToError(getMLIRDiagnostic(
              Error("hash '" + llvm::encodeBase64(regionHash.getHash()) +
                    "' could not be found in the cache"),
              r->getLoc()));
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
                mlir::ParserConfig(r->getContext(),
                                   /*verifyAfterParse=*/false)))) {
          return out.setToError(getMLIRDiagnostic(
              Error("reading bytecode file failed"), r->getLoc()));
        }

        // Get the container and take its body.
        ContainerOp container = cast<ContainerOp>(b.front());
        r->takeBody(container.getBodyRegion());

        // Finish up by replacing symbols/hashes with their original attrs.
        mlir::AttrTypeReplacer replacer;
        replacer.addReplacement([&](SymbolRefAttr symRef) -> SymbolRefAttr {
          size_t index;
          bool err =
              symRef.getLeafReference().getValue().getAsInteger(10, index);
          assert(!err && "Must have parsed the symbol ref as an integer!");
          return regionHash.getSymbols()[index];
        });
        replacer.addReplacement([&](HashIndexAttr hashRef) {
          return regionHash.getHashes()[hashRef.getIndex()];
        });
        r->walk([&](Operation *op) {
          replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                     /*replaceLocs=*/true,
                                     /*replaceTypes=*/true);
        });
        std::move(out).emplace();
      });

  return out;
}

AsyncValueRef<Chain> M::Cache::inflateOp(Operation *cached,
                                         RCRef<RegionCache> cache,
                                         AnyAsyncValueRef chain) {
  auto out = AsyncValueRef<Chain>::allocate(cache->getRuntime());

  // Hang the inflation off the input chain.
  std::move(chain).andThenSync([cached, cache = cache.copy(), out = out.copy()](
                                   AnyAsyncValueRef &&chain) mutable {
    if (chain.isError())
      return out.setToError(chain.takeDiagnostic());

    auto hashes =
        cached->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
    // If the op doesn't have any region hashes on it, we're done.
    if (!hashes)
      return std::move(out).emplace();

    // Fill in the regions on the operation.
    SmallVector<AnyAsyncValueRef> results;
    for (auto [regionHash, region] : llvm::zip(hashes, cached->getRegions()))
      results.push_back(inflateRegion(&region, regionHash, cache.copy()));

    // Once all the regions are cached, remove the region hash attr and
    // resolve success/failure.
    andThenSyncMoving(results,
                      [cached,
                       // Safe to move our copy of out.
                       out = std::move(out)](
                          MutableArrayRef<AnyAsyncValueRef> values) mutable {
                        for (auto &v : values)
                          if (v.isError())
                            return out.setToError(v.takeDiagnostic());

                        // Remove the region hash attr.
                        cached->removeAttr(getRegionHashAttrName());
                        // Done!
                        std::move(out).emplace();
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
