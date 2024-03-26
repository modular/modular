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
          out.getRuntime(), cacheAttr.getHash(),
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

/// Inflate a single region from `regionHash` and have `r` take its body.
static AsyncValueRef<Chain> inflateRegion(LLCL::Runtime &runtime, Region *r,
                                          RegionHashAttr regionHash,
                                          RCRef<RegionCache> cache) {
  auto out = AsyncValueRef<Chain>::allocate(runtime);

  auto foundOr =
      cache->find(runtime, regionHash.getHash(),
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
        auto container = readOpFromBytecodeFile<ContainerOp>(
            *bytecode, mlir::ParserConfig(r->getContext(),
                                          /*verifyAfterParse=*/false));
        if (!container) {
          return std::move(out).setToError(getMLIRDiagnostic(
              Error("reading bytecode file failed"), r->getLoc()));
        }
        r->takeBody(container->getBodyRegion());

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
  auto out = AsyncValueRef<Chain>::allocate(chain.getRuntime());

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
      results.push_back(
          inflateRegion(chain.getRuntime(), &region, regionHash, cache.copy()));

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
