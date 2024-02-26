//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_INTERPRETER_INTERPRETERDIALECT_H
#define KGEN_INTERPRETER_INTERPRETERDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/Threading/Shared.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterDialect.h.inc"

namespace M {

//===----------------------------------------------------------------------===//
// DialectResourceManager
//===----------------------------------------------------------------------===//

class MemoryHandle;

/// The IR resource manager for MDialect. It manages blob resource and tracks
/// which are pure string resources. This class may be accessed in a
/// multi-threaded context.
class DialectResourceManager
    : public mlir::DialectInterface::Base<DialectResourceManager> {
public:
  /// The kind of resource managed.
  enum class ResourceKind { Uninitialized, String, Blob };

  /// This class represents an individual resource entry.
  class ResourceEntry {
  public:
    /// Return the key used to reference this blob.
    StringRef getKey() const { return key; }

    /// Get the kind of resource.
    ResourceKind getKind() const { return kind; }

    /// Return the blob owned by this entry.
    const mlir::AsmResourceBlob *getBlob() const {
      return blob ? &*blob : nullptr;
    }
    mlir::AsmResourceBlob *getBlob() { return blob ? &*blob : nullptr; }

    /// Set the blob owned by this entry.
    void updateValue(ResourceKind newKind, mlir::AsmResourceBlob &&newBlob) {
      kind = newKind;
      blob = std::move(newBlob);
    }

  private:
    ResourceEntry() : kind(ResourceKind::Uninitialized) {}

    /// The key used for this blob.
    StringRef key;
    /// The kind of resource.
    ResourceKind kind;
    /// The blob that is referenced by this entry if it is valid.
    std::optional<mlir::AsmResourceBlob> blob;

    friend class DialectResourceManager;
  };

  DialectResourceManager(Dialect *dialect) : Base(dialect) {}

  /// Declare a new resource in the manager. This hook is invoked when parsing
  /// IR files to add resources to the dialect.
  MemoryHandle declareResource(StringRef key);

  /// Update the resource for the entry defined by the provided name with a
  /// blob encountered during parsing.
  void updateResourceWithBlob(StringRef key, mlir::AsmResourceBlob &&newBlob);
  /// Update the resource for the entry defined by the provided name with a
  /// string encountered during parsing.
  void updateResourceWithString(StringRef key, StringRef value);

  /// Get or add a blob resource with the provided data. String resources
  /// are deduplicated based on content.
  MemoryHandle getOrAddBlobResource(void *memory, size_t size, size_t align);
  /// Get or add a string resource with the provided data. String resources
  /// are deduplicated based on content.
  MemoryHandle getOrAddStringResource(StringRef value);

private:
  /// Wrap a resource entry into a handle.
  MemoryHandle createHandle(ResourceEntry *entry);

  /// Get or add a resource with the provided data. Resources are deduplicated
  /// based on content.
  MemoryHandle getOrAddResource(ArrayRef<char> data, size_t align,
                                ResourceKind kind);

  /// The internal map of tracked blobs. StringMap stores entries in distinct
  /// allocations, so we can freely take references to the data without fear of
  /// invalidation during additional insertion/deletion.
  Shared<llvm::StringMap<ResourceEntry>> resources;
};

//===----------------------------------------------------------------------===//
// MemoryHandle
//===----------------------------------------------------------------------===//

class MemoryHandle : public mlir::AsmDialectResourceHandleBase<
                         MemoryHandle, DialectResourceManager::ResourceEntry,
                         InterpreterDialect> {
public:
  using AsmDialectResourceHandleBase::AsmDialectResourceHandleBase;

  /// Return the human readable string key for this handle.
  StringRef getKey() const { return getResource()->getKey(); }

  /// Return the blob referenced by this handle if the underlying resource has
  /// been initialized. Returns nullptr otherwise.
  mlir::AsmResourceBlob *getBlob();
  const mlir::AsmResourceBlob *getBlob() const {
    return getResource()->getBlob();
  }

  /// Get the interface for the dialect that owns handles of this type. Asserts
  /// that the dialect is registered.
  static DialectResourceManager &getManagerInterface(MLIRContext *ctx);
};

} // namespace M

#endif // KGEN_INTERPRETER_INTERPRETERDIALECT_H
