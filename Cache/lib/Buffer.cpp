//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/Buffer.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;
using namespace Cache;

//===----------------------------------------------------------------------===//
// Buffer
//===----------------------------------------------------------------------===//

/// Create a Buffer from an array of data.
Buffer::Buffer(StringRef data) {
  kind = kMalloc;
  new (&this->mallocd) AllocatedBuffer(data);
}

Buffer::Buffer(llvm::sys::fs::mapped_file_region &&mapped) {
  kind = kMMap;
  new (&this->mapped) llvm::sys::fs::mapped_file_region(std::move(mapped));
}

Buffer::~Buffer() {
  switch (kind) {
  case kMalloc:
    this->mallocd.~AllocatedBuffer();
    break;
  case kMMap:
    this->mapped.~mapped_file_region();
    break;
  }
}

/// Open a file in read-only or read-write mode and return its file descriptor
/// and the status object for the file so we can get things like its size.
static ErrorOr<std::pair<llvm::sys::fs::file_t, llvm::sys::fs::file_status>>
openFile(const std::filesystem::path &filepath, bool readOnly) {
  std::string filepathStr = filepath.string();

  llvm::sys::fs::file_t fd;
  // llvm::Expected doesn't have a default constructor...so we have to duplicate
  // some code.
  if (readOnly) {
    llvm::Expected<llvm::sys::fs::file_t> fdOr =
        llvm::sys::fs::openNativeFileForRead(filepathStr,
                                             llvm::sys::fs::OF_None);
    if (!fdOr)
      return Error(toString(fdOr.takeError()));
    fd = *fdOr;
  } else {
    llvm::Expected<llvm::sys::fs::file_t> fdOr =
        llvm::sys::fs::openNativeFileForReadWrite(
            filepathStr, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::OF_None);
    if (!fdOr)
      return Error(toString(fdOr.takeError()));
    fd = *fdOr;
  }

  llvm::sys::fs::file_status status;
  if (std::error_code err = llvm::sys::fs::status(fd, status))
    return Error(err.message());

  // If this not a file or a block device (e.g. it's a named pipe
  // or character device), we can't mmap it, so error out.
  llvm::sys::fs::file_type type = status.type();
  if (type != llvm::sys::fs::file_type::regular_file &&
      type != llvm::sys::fs::file_type::block_file)
    return Error("cannot map file that is not an actual file or block device");

  return std::make_pair(fd, status);
}

ErrorOr<BufferRef> Buffer::getFile(const std::filesystem::path &filepath,
                                   size_t size, size_t offset) {
  auto fdOr = openFile(filepath, /*readOnly=*/true);
  if (fdOr.isError())
    return fdOr.takeError();
  llvm::sys::fs::file_t fd = fdOr->first;
  llvm::sys::fs::file_status status = fdOr->second;

  // If no size was provided, use the file's size.
  if (size == 0)
    size = status.getSize();

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readonly, size, offset, ec);
  if (ec)
    return Error(ec.message());

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return BufferRef::create(std::move(mappedFile));
}

const char *Buffer::getBufferStart() const {
  switch (kind) {
  case kMalloc:
    return (char *)this->mallocd.data;
  case kMMap:
    return this->mapped.const_data();
  default:
    llvm_unreachable("unknown buffer kind");
  }
}

const char *Buffer::getBufferEnd() const {
  switch (kind) {
  case kMalloc:
    return (char *)this->mallocd.data + this->mallocd.size;
  case kMMap:
    return this->mapped.const_data() + this->mapped.size();
  default:
    llvm_unreachable("unknown buffer kind");
  }
}

size_t Buffer::getBufferSize() const {
  switch (kind) {
  case kMalloc:
    return this->mallocd.size;
  case kMMap:
    return this->mapped.size();
  default:
    llvm_unreachable("unknown buffer kind");
  }
}

StringRef Buffer::getBuffer() const {
  return StringRef(getBufferStart(), getBufferSize());
}

//===----------------------------------------------------------------------===//
// Buffer::AllocatedBuffer
//===----------------------------------------------------------------------===//

Buffer::AllocatedBuffer::AllocatedBuffer(StringRef str) {
  data = malloc(str.size());
  assert(data && "malloc failed!");
  size = str.size();
  memcpy(data, str.begin(), size);
  align = sizeof(void *);
}

//===----------------------------------------------------------------------===//
// WriteableBuffer getFile
//===----------------------------------------------------------------------===//

ErrorOr<WriteableBufferRef>
WriteableBuffer::getFile(const std::filesystem::path &filepath, size_t size,
                         size_t offset) {
  auto fdOr = openFile(filepath, /*readOnly=*/false);
  if (fdOr.isError())
    return fdOr.takeError();
  llvm::sys::fs::file_t fd = fdOr->first;
  llvm::sys::fs::file_status status = fdOr->second;

  // Handle the size. If no size was provided, use the file's size. Otherwise,
  // resize the file before we map it in.
  if (size == 0) {
    size = status.getSize();
  } else if (status.getSize() < size) {
    // On Windows, the resize_file_before_mapping_readwrite is a no-op which
    // takes an integer file handle (and not an llvm::fs::file_t). To avoid
    // compilation failure, we just skip calling the
    // resize_file_before_mapping_readwrite function.
#ifndef _WIN32
    if (auto err =
            llvm::sys::fs::resize_file_before_mapping_readwrite(fd, size))
      return Error(err.message());
#endif // _WIN32
  }

  std::error_code ec;
  llvm::sys::fs::mapped_file_region mappedFile(
      fd, llvm::sys::fs::mapped_file_region::readwrite, size, offset, ec);
  if (ec)
    return Error(ec.message());

  // Close the file, mmap will hold a ref to the descriptor.
  llvm::sys::fs::closeFile(fd);

  return WriteableBufferRef::create(std::move(mappedFile));
}

//===----------------------------------------------------------------------===//
// WriteableBuffer raw_pwrite_stream implementation
//===----------------------------------------------------------------------===//

void WriteableBuffer::write_impl(const char *ptr, size_t size) {
  assert(kind == kMalloc && "cannot write to an mmap'd file");
  // We don't have an aligned realloc, so allocate an aligned buffer.
  void *tmp = alignedAlloc(this->mallocd.align, this->mallocd.size + size);
  assert(tmp && "alignedAlloc failed");
  // Copy the data in mallocd into tmp.
  memcpy(tmp, this->mallocd.data, this->mallocd.size);
  // Free the old thing now we've copied the data over.
  alignedFree(this->mallocd.data);
  // Set mallocd to tmp to complete the 'realloc'
  this->mallocd.data = tmp;
  // Finally, copy the new data in.
  memcpy((char *)this->mallocd.data + this->mallocd.size, ptr, size);
  this->mallocd.size += size;
}

// This implementation is essentially translated from the implementation of
// llvm::raw_svector_ostream.
void WriteableBuffer::pwrite_impl(const char *ptr, size_t size,
                                  uint64_t offset) {
  // TODO: currently we don't resize the mmap'd buffer.
  assert(getBufferStart() + offset + size <= getBufferEnd() || kind == kMalloc);
  if (kind == kMMap) {
    memcpy(this->mapped.data() + offset, ptr, size);
    return;
  }

  // Check how many bytes would be left over if we copied `size` bytes from
  // `ptr` into `offset`. The number of bytes written to the range is logically
  // offset + size. The number of bytes past the end will be getBufferSize() -
  // (offset + size). If that number is negative, then we're writing more bytes
  // than the buffer has.
  int64_t overflowBytes = (int64_t)getBufferSize() - (int64_t)(offset + size);
  if (overflowBytes >= 0) {
    memcpy((char *)this->mallocd.data + offset, ptr, size);
    return;
  }
  // Set it to positive and assert that it is in fact positive, and less than
  // size.
  int64_t leftoverBytes = -overflowBytes;
  assert(leftoverBytes > 0 && leftoverBytes < (int64_t)size &&
         "invalid leftover bytes somehow");

  // pwrite whatever won't overflow - the number of bytes until we get to the
  // end of the buffer.
  memcpy((char *)this->mallocd.data + offset, ptr, size - leftoverBytes);
  // And write the leftovers to the end.
  write_impl(ptr + (size - leftoverBytes), leftoverBytes);
}
