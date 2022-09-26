//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BlobCache.h"
#include "Support/HMAC.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include <filesystem>

using namespace M;

//===----------------------------------------------------------------------===//
// BlobCacheBackend
//===----------------------------------------------------------------------===//

ErrorOrSuccess BlobCacheBackend::insert(StringRef keyHash,
                                        llvm::MemoryBufferRef obj) {
  if (auto err = insertImpl(keyHash, obj))
    return err.takeError();

  if (delegate)
    if (auto err = delegate->insert(keyHash, obj))
      return err.takeError();

  return success();
}

bool BlobCacheBackend::contains(StringRef keyHash) {
  if (containsImpl(keyHash))
    return true;
  if (delegate)
    return delegate->contains(keyHash);
  return false;
}

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
BlobCacheBackend::find(StringRef keyHash) {
  if (containsImpl(keyHash))
    return this->findImpl(keyHash);

  if (!delegate)
    return Error("could not find item '" + keyHash + "'");

  auto itemOr = delegate->find(keyHash);
  if (failed(itemOr))
    return itemOr.takeError();

  std::unique_ptr<llvm::MemoryBuffer> item = std::move(*itemOr);
  // Store the item in our cache level so we can get a cache hit later.
  if (auto err = insertImpl(keyHash, *item))
    return err.takeError();

  // Return the item.
  return item;
}

//===----------------------------------------------------------------------===//
// InMemoryBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides an in-memory backend that stores memory buffers in an
/// llvm::StringMap.
struct InMemoryBackend : public BlobCacheBackend {
  ErrorOrSuccess insertImpl(StringRef keyHash,
                            llvm::MemoryBufferRef obj) override {
    // Store the item in this cache.
    cache[keyHash] = llvm::MemoryBuffer::getMemBufferCopy(
        obj.getBuffer(), obj.getBufferIdentifier());
    return success();
  }

  bool containsImpl(StringRef keyHash) const override {
    return cache.count(keyHash);
  }

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  findImpl(StringRef keyHash) const override {
    auto found = cache.find(keyHash);
    if (found == cache.end())
      return Error("could not find item '" + keyHash + "'");

    // Create a memory buffer that aliases this same data.
    auto &mbuf = (*found).second;
    return llvm::MemoryBuffer::getMemBuffer(mbuf->getBuffer(),
                                            mbuf->getBufferIdentifier());
  }

  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cache;
};
} // namespace

std::unique_ptr<BlobCacheBackend> M::getInMemoryBackend() {
  return std::make_unique<InMemoryBackend>();
}

//===----------------------------------------------------------------------===//
// FilesystemBackend
//===----------------------------------------------------------------------===//

namespace {
/// Provides a filesystem-backed backend that stores the buffers in binary files
/// on disk.
struct FilesystemBackend : public BlobCacheBackend {
  explicit FilesystemBackend(StringRef basePath) : basePath(basePath.str()) {}

  ErrorOrSuccess insertImpl(StringRef keyHash,
                            llvm::MemoryBufferRef obj) override {
    // Get the absolute path and create any directories we need to create.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    std::error_code err;
    std::filesystem::create_directories(filePath.parent_path(), err);
    if (err)
      return Error(err.message());

    // If the file doesn't exist, just touch and close immediately. We resize in
    // the next step, and then mmap it in as a writable buffer.
    if (!containsImpl(filePath.string().c_str()))
      fclose(fopen(filePath.string().c_str(), "w"));

    // Resize the file to contain enough bytes.
    std::filesystem::resize_file(filePath, obj.getBufferSize() + sha256Bytes,
                                 err);
    if (err)
      return Error(err.message());

    auto fileOr = llvm::WritableMemoryBuffer::getFile(filePath.string());
    if (!fileOr)
      return Error(fileOr.getError().message());

    std::unique_ptr<llvm::WritableMemoryBuffer> fileBuf = std::move(*fileOr);
    MutableArrayRef<char> buf = fileBuf->getBuffer();
    // Copy the data into the file buffer.
    auto outputIter =
        std::copy(obj.getBufferStart(), obj.getBufferEnd(), buf.begin());
    // And compute the HMAC and copy that in as well.
    SHA256Hash hash = hmacSHA256(obj.getBuffer(), kIntegrityKey);
    std::copy(hash.begin(), hash.end(), outputIter);
    return success();
  }

  bool containsImpl(StringRef keyHash) const override {
    auto abs = getAbsolutePathForKey(keyHash);
    return std::filesystem::exists(abs) && !std::filesystem::is_directory(abs);
  }

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  findImpl(StringRef keyHash) const override {
    // Get the file path and open it.
    std::filesystem::path filePath = getAbsolutePathForKey(keyHash);
    auto fileOr = llvm::WritableMemoryBuffer::getFile(filePath.string());

    // If the file doesn't exist, or it's empty, return an error.
    if (!fileOr)
      return Error(fileOr.getError().message());
    if ((*fileOr)->getBufferSize() == 0)
      return Error("file '" + filePath.string() + "' exists, but is empty");

    std::unique_ptr<llvm::MemoryBuffer> fileBuf = std::move(*fileOr);

    StringRef contentsAndHMAC = fileBuf->getBuffer();

    // Get a StringRef of the contents without the HMAC.
    StringRef contents(contentsAndHMAC.begin(),
                       contentsAndHMAC.size() - sha256Bytes);
    SHA256Hash hmac = hmacSHA256(contents, kIntegrityKey);
    auto storedHMACRange = llvm::make_range(
        contentsAndHMAC.begin() + contents.size(), contentsAndHMAC.end());

    // Check the computed hmac against the one in the file. This is a
    // constant-time memcmp.
    bool isIncorrect = false;
    for (auto [computed, stored] : llvm::zip(storedHMACRange, hmac))
      isIncorrect |= computed ^ stored;

    if (isIncorrect)
      return Error("corrupted file, stored hash and computed hash did not "
                   "match for file '" +
                   filePath.string() + "'");

    return fileBuf;
  }

  std::filesystem::path getAbsolutePathForKey(StringRef keyHash) const {
    std::filesystem::path filepath(basePath);
    filepath /= keyHash.str();

    return std::filesystem::absolute(filepath);
  }

  /// This is a CSPRNG-generated 32-byte string. It's used for integrity
  /// checking in the HMAC.
  static constexpr llvm::StringLiteral kIntegrityKey =
      "bedcaea9f09fa9fe565a8088ea66547c06c7c8e9c47fa46e0fb768a157d640a6";
  /// The base path for the filesystem cache.
  std::string basePath;
};
} // namespace

std::unique_ptr<BlobCacheBackend> M::getFilesystemBackend(StringRef basePath) {
  return std::make_unique<FilesystemBackend>(basePath);
}

std::unique_ptr<BlobCacheBackend>
M::getDefaultBackendChain(StringRef basePath) {
  auto backend = getInMemoryBackend();
  backend->setDelegate(getFilesystemBackend(basePath));
  return backend;
}
