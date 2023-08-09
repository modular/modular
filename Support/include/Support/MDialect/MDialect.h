//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MDIALECT_MDIALECT_H
#define SUPPORT_MDIALECT_MDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/Threading/Shared.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MDialect.h.inc"

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

  /// Copy the provided data into a new heap-allocated blob with the provided
  /// size and alignment. The base name will be used to form the key. If there
  /// is a conflict, the base name is automatically renamed.
  MemoryHandle addBlobResource(StringRef baseName, void *memory, size_t size,
                               size_t align);

private:
  /// The internal map of tracked blobs. StringMap stores entries in distinct
  /// allocations, so we can freely take references to the data without fear of
  /// invalidation during additional insertion/deletion.
  Shared<llvm::StringMap<ResourceEntry>> resources;
};

//===----------------------------------------------------------------------===//
// MemoryHandle
//===----------------------------------------------------------------------===//

class MemoryHandle
    : public mlir::AsmDialectResourceHandleBase<
          MemoryHandle, DialectResourceManager::ResourceEntry, MDialect> {
public:
  using AsmDialectResourceHandleBase::AsmDialectResourceHandleBase;

  /// Return the human readable string key for this handle.
  StringRef getKey() const { return getResource()->getKey(); }

  /// Return the blob referenced by this handle if the underlying resource has
  /// been initialized. Returns nullptr otherwise.
  mlir::AsmResourceBlob *getBlob() { return getResource()->getBlob(); }
  const mlir::AsmResourceBlob *getBlob() const {
    return getResource()->getBlob();
  }

  /// Get the interface for the dialect that owns handles of this type. Asserts
  /// that the dialect is registered.
  static DialectResourceManager &getManagerInterface(MLIRContext *ctx);
};

} // namespace M

#endif // SUPPORT_MDIALECT_MDIALECT_H
