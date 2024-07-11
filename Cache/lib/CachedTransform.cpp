//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachedTransform.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace Cache;
using namespace AsyncRT;

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

std::string TransformCacheKey::hashKey(TransformCacheKey::KeyTy key) {
  TimeTraceScope scope(
      CacheProfilerEntry::create("TransformCacheKey::hashKey"));

  // Reserve a 16 byte result hash.
  std::string result;
  result.reserve(sizeof(llvm::XXH128_hash_t));

  // Take the 128-bit xxhash of the input.
  llvm::XXH128_hash_t hash =
      llvm::xxh3_128bits(arrayRefFromStringRef(key->getBuffer()));

  // Write the hash to the result buffer and return it.
  result.append(llvm::bit_cast<char *>(&hash), sizeof(llvm::XXH128_hash_t));
  return result;
}

//===----------------------------------------------------------------------===//
// Operation Transformations
//===----------------------------------------------------------------------===//

LogicalResult Cache::writeOperationToCacheKey(Operation *op,
                                              const WriteableBufferRef &key) {
  TimeTraceScope scope(CacheProfilerEntry::create("writeOperationToCacheKey"));
  // Use bytecode when writing cache keys to ensure determinism across different
  // builds.
  return mlir::writeBytecodeToFile(op, *key);
}

/// Encode the given set of diagnostics in the provided buffer.
template <typename T>
static void encodeDiagnostics(T &&diagnostics, WriteableBufferRef buf) {
  llvm::support::endian::Writer writer(*buf, llvm::endianness::little);

  // Functor used to encode a string.
  auto encodeString = [&](StringRef str) {
    writer.write((uint64_t)str.size());
    *buf << str;
    buf->write((char)0);
  };

  // Write out the diagnostics.
  writer.write((uint64_t)llvm::size(diagnostics));
  for (Diagnostic &diag : diagnostics) {
    buf->write((char)diag.getSeverity());
    encodeString(mlir::debugString(diag.getLocation()));
    encodeString(diag.str());
    encodeDiagnostics(diag.getNotes(), buf.copy());
  }
}

/// Decode a set of diagnostics from the provided data. The provided data
/// pointer is updated to point to the next byte after the diagnostics.
static ErrorOrSuccess decodeDiagnostics(const char *&dataIt,
                                        const char *dataEnd, MLIRContext *ctx,
                                        std::vector<Diagnostic> &diagnostics) {
  // Functor for reading a uint64_t from the cache buffer.
  auto readInt = [&](uint64_t &value) -> ErrorOrSuccess {
    if ((dataIt + sizeof(uint64_t)) > dataEnd)
      return Error("failed to read int from cache buffer");
    value = llvm::support::endian::readNext<uint64_t, llvm::endianness::little,
                                            llvm::support::unaligned>(dataIt);
    return success();
  };

  // Functor for reading a string from the cache buffer.
  auto readString = [&](StringRef &str) -> ErrorOrSuccess {
    uint64_t size = 0;
    if (auto err = readInt(size))
      return err.takeError();
    if ((dataIt + size + 1) > dataEnd)
      return Error("failed to read string from cache buffer");
    str = StringRef(dataIt, size);
    dataIt += (size + 1);
    return success();
  };

  // Write out the number of diagnostics.
  uint64_t numDiagnostics = 0;
  if (auto err = readInt(numDiagnostics))
    return err;
  for (uint64_t i = 0; i < numDiagnostics; ++i) {
    if (dataIt == dataEnd)
      return Error("failed to read diagnostic from cache buffer");
    char severity = *dataIt++;

    // Read in the location.
    StringRef locationStr;
    if (auto err = readString(locationStr))
      return err;
    LocationAttr loc = dyn_cast_if_present<LocationAttr>(
        mlir::parseAttribute(locationStr, ctx, Type(), /*numRead=*/nullptr,
                             /*isKnownNullTerminated=*/true));
    if (!loc)
      return Error("failed to parse location in cached diagnostic");
    mlir::Diagnostic diag(loc, static_cast<mlir::DiagnosticSeverity>(severity));

    // Read in the message.
    StringRef message;
    if (auto err = readString(message))
      return err;
    diag << message;

    // Read in the notes.
    std::vector<Diagnostic> notes;
    if (auto err = decodeDiagnostics(dataIt, dataEnd, ctx, notes))
      return err;
    for (Diagnostic &note : notes)
      diag.attachNote() = std::move(note);

    diagnostics.push_back(std::move(diag));
  }
  return success();
}
static ErrorOrSuccess decodeDiagnostics(StringRef &data, MLIRContext *ctx,
                                        std::vector<Diagnostic> &diagnostics) {
  const char *dataIt = data.data();
  if (auto err = decodeDiagnostics(dataIt, data.end(), ctx, diagnostics))
    return err;

  // Update the data string.
  data = StringRef(dataIt, data.end() - dataIt);
  return success();
}

/// Create a deep copy of the given diagnostic.
static Diagnostic copyDiag(const Diagnostic &diag) {
  Diagnostic newDiag(diag.getLocation(), diag.getSeverity());
  newDiag << diag.str();
  for (auto &note : diag.getNotes())
    newDiag.attachNote() = copyDiag(note);
  return newDiag;
}

/// Run a pass manager's passes as a cached transform.
AnyAsyncValueRef
Cache::cachedTransform(Operation *target, RCRef<TransformCache> transformCache,
                       AnyAsyncValueRef chain, mlir::PassManager &pm,
                       const std::function<void(Operation *)> &moreOnMiss,
                       const std::function<void(Operation *)> &moreOnHit) {
  auto keyBuf = WriteableBuffer::get();
  pm.printAsTextualPipeline(*keyBuf);

  // Callback that runs the pass manager and puts the correct region hash attr
  // on the op.
  auto runTransform =
      [&pm, moreOnMiss](Operation *op, WriteableBufferRef buf,
                        AnyAsyncValueRef chain) -> AsyncValueRef<Chain> {
    TimeTraceScope traceScope(CacheProfilerEntry::create(
        "Cache::cachedTransform(Operation *)::runTransform"));
    // Allocate a space to put the result of the pass manager (the emitted
    // diagnostics). We'll chain off that for the deflation.
    auto pmResult =
        AsyncValueRef<std::vector<Diagnostic>>::allocate(chain.getRuntime());
    std::move(chain).andThenSync([op, &pm, moreOnMiss,
                                  pmResult = pmResult.copy()](
                                     AnyAsyncValueRef &&chain) mutable {
      moreOnMiss(op);

      if (chain.isError())
        return std::move(pmResult).setToError(chain.takeDiagnostic());

      // Collect the diagnostics emitted while running the pass manager. These
      // will get cached with the bytecode.
      std::vector<Diagnostic> diagnostics;
      auto handlerFn = [&](const Diagnostic &diag) {
        diagnostics.push_back(copyDiag(diag));

        // Return failure to allow the main handler to process the diagnostic.
        return failure();
      };
      mlir::ScopedDiagnosticHandler diagHandler(op->getContext(), handlerFn);

      if (failed(pm.run(op))) {
        return std::move(pmResult).setToError(getMLIRDiagnostic(
            Error("failed to run the pass manager"), op->getLoc()));
      }

      std::move(pmResult).emplace(std::move(diagnostics));
    });

    auto out = AsyncValueRef<Chain>::allocate(pmResult.getRuntime());
    // Just write the bytecode and return.
    std::move(pmResult).andThenSync(
        [op, buf = std::move(buf), out = out.copy()](
            AsyncValueRef<std::vector<Diagnostic>> &&pmResult) mutable {
          if (pmResult.isError())
            return std::move(out).setToError(pmResult.takeDiagnostic());

          // Write out the bytecode.
          TimeTraceScope traceScope(
              CacheProfilerEntry::create("writeBytecodeToFile"));
          if (failed(mlir::writeBytecodeToFile(op, *buf))) {
            return std::move(out).setToError(getMLIRDiagnostic(
                "failed to write bytecode file", op->getLoc()));
          }

          // Write out the diagnostics.
          uint64_t bufSize = buf->getBuffer().size();
          encodeDiagnostics(*pmResult, buf.copy());

          // Write out the bytecode size as a footer so that we can step
          // over it when reading the diagnostics. We encode this at the end
          // to make sure that the bytecode is still aligned.
          llvm::support::endian::Writer(*buf, llvm::endianness::little)
              .write(bufSize);

          std::move(out).emplace();
        });
    return out;
  };

  // Callback that on a cache hit reads the region hashes out of the cache and
  // places them on the operation.
  auto onCacheHit = [moreOnHit](Operation *op,
                                const BufferRef &buf) -> ErrorOrSuccess {
    moreOnHit(op);
    TimeTraceScope traceScope(CacheProfilerEntry::create(
        "Cache::cachedTransform(Operation *)::onCacheHit"));
    StringRef buffer = buf->getBuffer();
    MLIRContext *ctx = op->getContext();

    // Decode the cached diagnostics and re-emit them.
    auto readDiagnostics = [ctx](StringRef &diagBuffer) -> ErrorOrSuccess {
      std::vector<Diagnostic> diagnostics;
      if (auto err = decodeDiagnostics(diagBuffer, ctx, diagnostics))
        return err;
      for (Diagnostic &diag : diagnostics)
        ctx->getDiagEngine().emit(std::move(diag));
      return success();
    };

    // Read in the cached diagnostics encoded after the bytecode. The footer
    // of the buffer contains the size of the bytecode section.
    uint64_t bytecodeSize =
        llvm::support::endian::read<uint64_t, llvm::endianness::little,
                                    llvm::support::unaligned>(buffer.end() -
                                                              sizeof(uint64_t));
    StringRef bytecodeBuffer = buffer.take_front(bytecodeSize);
    buffer = buffer.drop_front(bytecodeSize);

    // Read in the encoded diagnostics.
    if (auto err = readDiagnostics(buffer))
      return err;

    // Parse the bytecode.
    std::unique_ptr<llvm::MemoryBuffer> bytecode =
        llvm::MemoryBuffer::getMemBuffer(bytecodeBuffer, /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);
    OwningOpRef<Operation *> cachedOp = readOpFromBytecodeFile(
        *bytecode, mlir::ParserConfig(op->getContext(),
                                      /*verifyAfterParse=*/false));

    // Get the body from the parsed op and onto the op we're using.
    for (auto [cached, opRegion] :
         llvm::zip(cachedOp->getRegions(), op->getRegions()))
      opRegion.takeBody(cached);

    return success();
  };

  return cachedTransform(target, std::move(transformCache), std::move(chain),
                         std::move(keyBuf), std::move(runTransform),
                         std::move(onCacheHit));
}
