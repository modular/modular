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
#include "Support/MDialect/MAttrs.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SetVector.h"
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
      return std::move(out).setToError(chain.takeDiagnostic());

    // Use the replacer strategy to replace "large" attributes with the
    // hashed version.
    mlir::AttrTypeReplacer replacer;
    // For now, we only care about DenseResourceElementsAttr because that's
    // how we handle large attributes.
    replacer.addReplacement([&](DenseResourceElementsAttr resourceAttr)
                                -> Attribute {
      mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
      // If the blob isn't there, we shouldn't try caching nothing.
      if (!blob)
        return nullptr;

      BufferRef resourceData = Buffer::get(
          StringRef(blob->getData().data(), blob->getData().size()));

      // We have to make this synchronous for now :(
      auto contains = cache->contains(resourceAttr);
      await(contains);
      // Only do the insert if we don't already have it in the cache.
      std::string keyHash = cache->getHash(resourceAttr);
      if (!*contains) {
        // Insert the data into the cache. We don't really care about the
        // ordering of insert if we have 2 threads inserting the same data -
        // the result will be that the last one wins but since it's the same
        // data we still end up with the correct result. The `contains`
        // check is just for the obvious case.
        AsyncValueRef<std::string> hashOr = cache->insert(
            resourceAttr, std::move(resourceData),
            MLIRLocationDecoder::getEncodedLocation(constant->getLoc()));
        // This is not great - we have to make this sync because MLIR
        // doesn't really have a good way to handle async here.
        await(hashOr);
        if (hashOr.isError()) {
          std::move(out).setToError(hashOr.takeDiagnostic());
          return nullptr;
        }

        keyHash = *hashOr;
      }

      // Create a builder so we can create attrs easier.
      OpBuilder builder(constant);

      NamedAttrList additionalAttrs;
      // The resource attribute may include a type annotation which conveys
      // its alignment. Otherwise use the blob's alignment.
      uint64_t align = dyn_cast<HasAlignedBytesInterface>(resourceAttr)
                           .getAlignedBytesType()
                           .getAlign();
      additionalAttrs.set(
          "align",
          builder.getIntegerAttr(
              builder.getType<IntegerType>(64, IntegerType::Unsigned), align));
      additionalAttrs.set(
          "name", builder.getStringAttr(resourceAttr.getRawHandle().getKey()));

      auto newAttr = ConstantHashAttr::get(
          resourceAttr.getContext(), resourceAttr.getType(), keyHash,
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
      return std::move(out).setToError(chain.takeDiagnostic());

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
      auto found = cache->find(
          cacheAttr.getHash(),
          MLIRLocationDecoder::getEncodedLocation(constant->getLoc()));
      await(found);
      if (found.isError()) {
        std::move(out).setToError(found.takeDiagnostic());
        return nullptr;
      }
      if (!found->has_value()) {
        std::move(out).setToError(getMLIRDiagnostic(
            Error("hash '" + llvm::encodeBase64(cacheAttr.getHash()) +
                  "' could not be found in the cache"),
            constant->getLoc()));
        return nullptr;
      }

      // Pull out any attributes we might need.
      DictionaryAttr additional = cacheAttr.getAdditionalData();
      IntegerAttr alignAttr = cast<IntegerAttr>(additional.get("align"));
      StringAttr name = cast<StringAttr>(additional.get("name"));

      // The cache owns the data, so we can simply hold a ref inside
      // UnmanagedAsmResourceBlob and drop it when it's done.
      BufferRef buf = std::move(**found);
      auto blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
          ArrayRef<char>(buf->getBufferStart(), buf->getBufferSize()),
          alignAttr.getUInt(),
          [buf = buf.copy()](void *data, size_t size, size_t align) {
            ; // No-op, buffer ref is dropped when this closure dies.
          });
      auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
          constant->getContext());

      // Return the new DenseResourceElementsAttr.
      auto newAttr = DenseResourceElementsAttr::get(
          cast<ShapedType>(cacheAttr.getType()),
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
  llvm::BLAKE3 hashState;
  hashState.init();

  auto hashTypeOrAttr = [&](auto t) {
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << t;
    hashState.update(stringStream.str());
  };

  for (auto arg : r->getArgumentTypes())
    hashTypeOrAttr(arg);

  r->walk([&](Operation *op) {
    // Add the operation's name to the hash.
    hashState.update(op->getName().getStringRef());

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
          // If it's an attr that we want to replace with an index, we'll do the
          // replacement so don't hash it. Otherwise, do hash it - and this
          // includes the ReplaceableAttrIndex.
          if (isa<ReplaceableAttr>(attr))
            return;
          hashTypeOrAttr(attr);
        },
        [](Type) {});
  });

  // Finalize the hash.
  std::array<uint8_t, 32> hash = hashState.final();
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
  llvm::SetVector<Attribute> attrs;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](ReplaceableAttr attr) { attrs.insert(attr); });

  r.walk([&](Operation *op) {
    walker.walk(op->getAttrDictionary());
    for (auto t : op->getOperandTypes())
      walker.walk(t);
    for (auto t : op->getResultTypes())
      walker.walk(t);
  });

  // Walk all the ops and replace their symbol refs with symbol indices, and
  // their hash refs with hash indices.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ReplaceableAttr repl) {
    auto found = llvm::find(attrs, repl);
    assert(found != attrs.end());
    return repl.convertToIndex(std::distance(attrs.begin(), found));
  });

  r.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true,
                               /*replaceTypes=*/true);
  });

  // Finally, we can store the region. Create an op to hang it off of so we
  // can cache it.
  auto container = builder.create<ContainerOp>(r.getLoc(), r);

  // This function contains the logic to attach a provided hash to the op -
  // since we need it in a couple places we just outline it here.
  auto attachHash = [op, container,
                     attrs = std::move(attrs)](std::string &&hash) mutable {
    // Create a new builder because this may run well after the rest of
    // this function.
    OpBuilder builder(op);
    SmallVector<RegionHashAttr> hashVec;
    // If we already have some hashes, we have to append to the end of
    // that array.
    auto hashes =
        op->getAttrOfType<RegionHashArrayAttr>(getRegionHashAttrName());
    if (hashes)
      hashVec = SmallVector<RegionHashAttr>(hashes.begin(), hashes.end());

    hashVec.push_back(
        builder.getAttr<RegionHashAttr>(hash, attrs.getArrayRef()));

    auto hashVecAttr = builder.getAttr<RegionHashArrayAttr>(hashVec);
    op->setAttr(getRegionHashAttrName(), hashVecAttr);

    // Finally, erase the container.
    container.erase();
  };

  auto out = AsyncValueRef<Chain>::allocate(cache->getRuntime());
  // Store it, but only if we don't already have it.
  AsyncValueRef<bool> contains =
      cache->contains(&container.getBodyRegion(),
                      MLIRLocationDecoder::getEncodedLocation(op->getLoc()));
  std::move(contains).andThenSync(
      [container, attachHash = std::move(attachHash), out = out.copy(),
       cache = std::move(cache)](AsyncValueRef<bool> &&contains) mutable {
        if (contains.isError())
          return std::move(out).setToError(contains.takeDiagnostic());

        // If we have it, then attach the hash to the op and move along.
        if (*contains) {
          attachHash(cache->getHash(&container.getBodyRegion()));
          return std::move(out).emplace();
        }

        // We don't already have it - add it.
        // Create a place to store the bytecode.
        WriteableBufferRef bytecode = WriteableBuffer::get();
        // Store the container in bytecode.
        if (failed(mlir::writeBytecodeToFile(container, *bytecode))) {
          return std::move(out).setToError(getMLIRDiagnostic(
              "failed to write bytecode file", container.getLoc()));
        }
        AsyncValueRef<std::string> hashOr =
            cache->insert(&container.getBodyRegion(), std::move(bytecode));
        // Keeping references is safe here because all the memory is owned by
        // the MLIRContext, which is guaranteed to live longer than any of this.
        std::move(hashOr).andThenSync(
            [attachHash = std::move(attachHash),
             out = out.copy()](AsyncValueRef<std::string> &&hashOr) mutable {
              // Check for errors.
              if (hashOr.isError())
                return std::move(out).setToError(hashOr.takeDiagnostic());

              attachHash(std::move(*hashOr));
              return std::move(out).emplace();
            });
      });

  return out;
}

AsyncValueRef<Chain> M::Cache::deflateOp(Operation *op,
                                         RCRef<RegionCache> cache,
                                         AnyAsyncValueRef chain) {
  TimeTraceScope traceScope(CacheProfilerEntry::create("Cache::deflateOp"));
  auto out = AsyncValueRef<Chain>::allocate(chain.getRuntime());
  // Hang the actual deflation off the input chain. This will allow users to
  // not worry about sequencing w.r.t. this operation, they can just pass in
  // the chain.
  std::move(chain).andThenSync([op, cache = cache.copy(), out = out.copy()](
                                   AnyAsyncValueRef &&chain) mutable {
    if (chain.isError())
      return std::move(out).setToError(chain.takeDiagnostic());

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
              return std::move(out).setToError(v.takeDiagnostic());

          std::move(out).emplace();
        });
  });

  return out;
}

/// Inflate a single region from `regionHash` and have `r` take its body.
static AsyncValueRef<Chain> inflateRegion(Region *r, RegionHashAttr regionHash,
                                          RCRef<RegionCache> cache) {
  auto out = AsyncValueRef<Chain>::allocate(cache->getRuntime());

  auto foundOr =
      cache->find(regionHash.getHash(),
                  MLIRLocationDecoder::getEncodedLocation(r->getLoc()));
  std::move(foundOr).andThenSync(
      [r, regionHash, out = out.copy()](
          AsyncValueRef<std::optional<BufferRef>> &&foundOr) mutable {
        if (foundOr.isError()) {
          return std::move(out).setToError(foundOr.takeDiagnostic());
        }
        if (!foundOr->has_value()) {
          return std::move(out).setToError(getMLIRDiagnostic(
              Error("hash '" + llvm::encodeBase64(regionHash.getHash()) +
                    "' could not be found in the cache"),
              r->getLoc()));
        }

        // Parse the bytecode for the region.
        BufferRef bytecodeBuf = std::move(**foundOr);
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
          return std::move(out).setToError(getMLIRDiagnostic(
              Error("reading bytecode file failed"), r->getLoc()));
        }

        // Get the container and take its body.
        ContainerOp container = cast<ContainerOp>(b.front());
        r->takeBody(container.getBodyRegion());

        // Finish up by replacing symbols/hashes with their original attrs.
        mlir::AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](ReplaceableAttrIndex ref) -> ReplaceableAttr {
              return ref.convertFromIndex(regionHash.getParams());
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
  TimeTraceScope traceScope(CacheProfilerEntry::create("Cache::inflateOp"));
  auto out = AsyncValueRef<Chain>::allocate(cache->getRuntime());

  // Hang the inflation off the input chain.
  std::move(chain).andThenSync([cached, cache = cache.copy(), out = out.copy()](
                                   AnyAsyncValueRef &&chain) mutable {
    if (chain.isError())
      return std::move(out).setToError(chain.takeDiagnostic());

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
    andThenSyncMoving(
        results, [cached,
                  // Safe to move our copy of out.
                  out = std::move(out)](
                     MutableArrayRef<AnyAsyncValueRef> values) mutable {
          for (auto &v : values)
            if (v.isError())
              return std::move(out).setToError(v.takeDiagnostic());

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
