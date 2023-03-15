//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_BUFFER_H
#define CACHE_BUFFER_H

#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/AlignedAlloc.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace M::Cache {
class Buffer;
using BufferRef = LLCL::RCRef<Buffer>;

/// Provides a reference-counted version of an LLVM memory buffer that owns its
/// data. This is useful for caching where you might want to store a buffer
/// that can't be deallocated until it's been (asynchronously) stored in the
/// cache. Buffer is read-only, for writing one should use WriteableBuffer
/// (defined below).
// TODO: Should this hold a reference to a LLCL::Allocator and use that to
//       allocate memory?
class Buffer : public LLCL::ReferenceCounted<Buffer> {
public:
  /// Destroy a buffer. Releases any resources associated with that buffer.
  virtual ~Buffer() = default;

  /// Create a buffer from a StringRef of data. This will copy the data into the
  /// resulting BufferRef.
  static BufferRef get(StringRef data) { return BufferRef::create(data); }

  /// Map in a file and use it as the backing storage for the BufferRef. If size
  /// and offset are provided, then a sub-range of the file is mapped in. This
  /// file is mapped read-only.
  static ErrorOr<BufferRef> getFile(const std::filesystem::path &filepath,
                                    size_t size = 0, size_t offset = 0);

  //===-------------------------------------------------------------------===//
  // llvm::MemoryBuffer API
  //===-------------------------------------------------------------------===//

  /// Provide essentially the same API as llvm::MemoryBuffer.
  const char *getBufferStart() const;
  const char *getBufferEnd() const;
  size_t getBufferSize() const;
  StringRef getBuffer() const;

protected:
  /// So RCRef can access protected constructors.
  friend class LLCL::RCRef<Buffer>;

  /// Initialize an empty Buffer. This is protected because we don't want to
  /// initialize empty read-only buffers.
  Buffer() = default;

  /// Create a Buffer of given size and alignment.
  Buffer(size_t size, std::optional<size_t> alignment)
      : storage{AllocatedBuffer(size, alignment)} {}

  /// Construct the Buffer where it has to copy its data.
  Buffer(StringRef data) : storage{AllocatedBuffer(data)} {}

  /// Construct a buffer with a mapped file region. The buffer takes ownership
  /// of the mapped file region.
  Buffer(llvm::sys::fs::mapped_file_region &&mapped)
      : storage{std::move(mapped)} {}

  /// Buffers are not copy-constructible.
  Buffer(const Buffer &other) = delete;
  Buffer &operator=(const Buffer &other) = delete;

  /// Struct to hold the data we need if this is a malloc'd buffer.
  struct AllocatedBuffer {
    void *data = nullptr;
    size_t size = 0;
    size_t align = 0;

    /// Default-construct a mallocd buffer to nullptr/0 with the given
    /// alignment.
    AllocatedBuffer()
        : data(nullptr), size(0), align(alignof(std::max_align_t)) {}
    AllocatedBuffer(size_t size, std::optional<size_t> alignment)
        : data(nullptr), size(size),
          align(alignment.value_or(alignof(std::max_align_t))) {
      data = alignedAlloc(align, size);
    }
    ~AllocatedBuffer() { alignedFree(data); }

    /// Moving a MallocdBuffer should transfer ownership.
    AllocatedBuffer(AllocatedBuffer &&other)
        : data(other.data), size(other.size), align(other.align) {
      other.data = nullptr;
      other.size = 0;
      other.align = 0;
    }

    /// Move the other buffer into `this`.
    AllocatedBuffer &operator=(AllocatedBuffer &&other) {
      if (&other != this)
        new (this) AllocatedBuffer(std::move(other));

      return *this;
    }

    /// Construct a MallocdBuffer from a StringRef.
    AllocatedBuffer(StringRef str);
  };

  /// The data owned by this buffer.
  /// AllocatedBuffer is the first type so that the default constructor is an
  /// empty AllocatedBuffer.
  SmartVariant<AllocatedBuffer, llvm::sys::fs::mapped_file_region> storage;
};

class WriteableBuffer;
using WriteableBufferRef = LLCL::RCRef<WriteableBuffer>;

/// Subclass of Buffer that is write-able. It also owns its data in all cases.
class WriteableBuffer : public Buffer, public llvm::raw_pwrite_stream {
public:
  /// Initialize an empty WriteableBuffer that can be written to.
  WriteableBuffer() : Buffer() { SetUnbuffered(); }
  /// Create a WriteableBuffer with initial size (this sets both the capacity
  /// and the number of bytes stored in the buffer to `size`). The user can also
  /// provide an alignment for the underlying allocation.
  WriteableBuffer(size_t size, std::optional<size_t> alignment = std::nullopt)
      : Buffer(size, alignment) {
    SetUnbuffered();
  }

  static WriteableBufferRef get() { return WriteableBufferRef::create(); }
  /// Create a WriteableBuffer with initial size (this sets both the capacity
  /// and the number of bytes stored in the buffer to `size`). The user can also
  /// provide an alignment for the underlying allocation.
  static WriteableBufferRef
  get(size_t size, std::optional<size_t> alignment = std::nullopt) {
    return WriteableBufferRef::create(size, alignment);
  }

  /// Map in a file and use it as the backing storage for the BufferRef. If
  /// size and offset are provided, then a sub-range of the file is mapped in.
  /// This file is mapped read/write.
  static ErrorOr<WriteableBufferRef>
  getFile(const std::filesystem::path &filepath, size_t size = 0,
          size_t offset = 0);

  /// Keep all the reader APIs.
  using Buffer::getBuffer;
  using Buffer::getBufferEnd;
  using Buffer::getBufferSize;
  using Buffer::getBufferStart;

  char *getBufferStart() {
    return const_cast<char *>(Buffer::getBufferStart());
  }

  char *getBufferEnd() { return const_cast<char *>(Buffer::getBufferEnd()); }

  MutableArrayRef<char> getBuffer() {
    return {getBufferStart(), getBufferEnd()};
  }

  //===-------------------------------------------------------------------===//
  // raw_pwrite_stream implementation
  //===-------------------------------------------------------------------===//

  /// Copies `size` bytes from address `ptr` to the end of the buffer (this
  /// resizes the buffer to contain getBufferSize() + `size` bytes).
  void write_impl(const char *ptr, size_t size) override;
  uint64_t current_pos() const override { return getBufferSize(); }
  void pwrite_impl(const char *ptr, size_t size, uint64_t offset) override;

private:
  /// So RCRef can access protected constructors.
  friend class LLCL::RCRef<WriteableBuffer>;

  /// Construct a WriteableBuffer from a mapped file region, and make sure to
  /// set to unbuffered.
  WriteableBuffer(llvm::sys::fs::mapped_file_region &&mapped)
      : Buffer(std::move(mapped)) {
    SetUnbuffered();
  }
};
} // namespace M::Cache

#endif // CACHE_BUFFER_H
