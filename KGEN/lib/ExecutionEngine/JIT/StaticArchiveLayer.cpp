//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine/JIT/StaticArchiveLayer.h"

#include "Cache/Support/Keys.h"
#include "KGEN/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/Base64.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

//===----------------------------------------------------------------------===//
// StaticArchiveObjectMaterializationUnit
//===----------------------------------------------------------------------===//

namespace {
class StaticArchiveObjectMaterializationUnit
    : public llvm::orc::MaterializationUnit {
public:
  StaticArchiveObjectMaterializationUnit(llvm::orc::ObjectLayer &objLayer,
                                         llvm::MemoryBufferRef objectBuffer,
                                         Interface &interface)
      : MaterializationUnit(interface), objectBuffer(objectBuffer),
        genLayer(objLayer) {}

  /// Provide a name for this MU that will show up in ORC debug logs.
  StringRef getName() const override {
    return "KGEN::StaticArchiveObjectMaterializationUnit";
  }

  /// Given a MaterializationResponsibility, push the object file buffer onto
  /// the base layer.
  void materialize(
      std::unique_ptr<llvm::orc::MaterializationResponsibility> mr) override {
    genLayer.emit(std::move(mr),
                  llvm::MemoryBuffer::getMemBuffer(
                      objectBuffer, /*RequiresNullTerminator=*/false));
  }

  /// Notify that the symbol `name` has been overridden.
  void discard(const llvm::orc::JITDylib &jd,
               const llvm::orc::SymbolStringPtr &name) override {}

  llvm::MemoryBufferRef objectBuffer;
  llvm::orc::ObjectLayer &genLayer;
};
} // namespace

//===----------------------------------------------------------------------===//
// StaticArchiveMaterializationLayer
//===----------------------------------------------------------------------===//

StaticArchiveLayer::StaticArchiveLayer(llvm::orc::ObjectLayer &objLayer,
                                       llvm::orc::ExecutionSession &sess,
                                       const llvm::DataLayout &dl,
                                       AddToSearchOrderFn add)
    : MaterializationLayer(LayerKind::kStaticArchiveLayer, sess, dl,
                           std::move(add)),
      objectLayer(objLayer) {}

ErrorOrSuccess StaticArchiveLayer::add(StringRef libName, BufferRef archive) {
  auto dylibOr = getOrCreateDylib(libName);
  if (dylibOr.isError())
    return dylibOr.takeError();
  llvm::orc::JITDylib *dylib = *dylibOr;

  // If the archive creation succeeds we store a ref to this buffer so the
  // data won't be deallocated until the JIT is destroyed. This version of
  // MemoryBuffer::getMemBuffer produces a non-owning buffer.
  std::unique_ptr<llvm::MemoryBuffer> archiveMemBuf =
      llvm::MemoryBuffer::getMemBuffer(archive->getBuffer(),
                                       /*BufferName=*/"",
                                       /*RequiresNullTerminator=*/false);
  auto archiveBinary = toModularErrorOr(
      llvm::object::Archive::create(archiveMemBuf->getMemBufferRef()));
  if (archiveBinary.isError())
    return archiveBinary.takeError();

  // Store a ref to the buffer data.
  archiveBuffers.push_back(archive.copy());

  // Generate a materialization unit for each of the children in this archive.
  // TODO: We really shouldn't have to do this, we should be able to use a
  // static library generator instead. This unfortunately doesn't work well with
  // the current generator model in orc, where some platforms (like MSVC) define
  // "terminal" generators as part of platform setup.
  llvm::orc::ResourceTrackerSP resourceTracker =
      dylib->getDefaultResourceTracker();
  llvm::Error err = llvm::Error::success();
  for (auto &child : (*archiveBinary)->children(err)) {
    if (err)
      return toModularError(std::move(err));
    auto childBufferOr = child.getMemoryBufferRef();
    if (!childBufferOr)
      return M::Error(toString(childBufferOr.takeError()));

    auto childInterface = toModularErrorOr(
        llvm::orc::getObjectFileInterface(session, *childBufferOr));
    if (childInterface.isError())
      return childInterface.takeError();
    if (auto defineErr = toModularErrorOr(dylib->define(
            std::make_unique<StaticArchiveObjectMaterializationUnit>(
                objectLayer, *childBufferOr, *childInterface),
            resourceTracker)))
      return defineErr;
  }
  if (err)
    return toModularError(std::move(err));

  return success();
}
