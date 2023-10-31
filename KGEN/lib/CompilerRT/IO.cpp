//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "Support/AlignedAlloc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace M;

static llvm::StringRef copyBytes(llvm::ArrayRef<char> data,
                                 size_t alignment = kPreferredMemoryAlignment) {
  char *ptr = reinterpret_cast<char *>(alignedAlloc(alignment, data.size()));
  llvm::copy(data, ptr);
  return {ptr, data.size()};
}

static llvm::StringRef copyString(llvm::StringRef str) {
  char *ptr = reinterpret_cast<char *>(
      alignedAlloc(kPreferredMemoryAlignment, str.size() + 1));
  memcpy(ptr, str.data(), str.size());
  ptr[str.size()] = '\0';
  return llvm::StringRef(ptr, str.size());
}

namespace {
struct FileHandle {

  static FileHandle *get(llvm::StringRef path, llvm::StringRef mode,
                         llvm::StringRef *errMsg) {
    std::filesystem::path fsPath(path.str());
    llvm::sys::fs::FileAccess fileAccess = getFileAccess(mode);
    if (fileAccess & llvm::sys::fs::FileAccess::FA_Write) {

      // TODO: When `append` mode is supported account for that.
      std::error_code err = llvm::sys::fs::remove(fsPath.string());
      if (err) {
        *errMsg = copyString(
            (llvm::Twine("unable to remove existing file ") + fsPath.string())
                .str());
        return nullptr;
      }

      std::filesystem::create_directories(fsPath.parent_path(), err);
      if (err) {
        *errMsg = copyString((llvm::Twine("unable to create directories '") +
                              fsPath.parent_path().string() + "' for write")
                                 .str());
        return nullptr;
      }
    } else if (fileAccess & llvm::sys::fs::FileAccess::FA_Read &&
               !std::filesystem::exists(fsPath)) {
      *errMsg = copyString(
          (llvm::Twine("file path '") + path + "' not found for read").str());
      return nullptr;
    }

    int handle;
    std::error_code err = llvm::sys::fs::openFile(
        path, handle, llvm::sys::fs::CreationDisposition::CD_OpenAlways,
        fileAccess, llvm::sys::fs::OpenFlags::OF_None);

    if (err) {
      *errMsg = copyString(err.message());
      return nullptr;
    }

    return new FileHandle(handle);
  }

  uint64_t size(llvm::StringRef *errMsg) {
    llvm::sys::fs::file_status status;
    if (std::error_code err = llvm::sys::fs::status(handle, status)) {
      *errMsg = copyString(err.message());
      return 0;
    }

    return status.getSize();
  }

  llvm::StringRef read(llvm::StringRef *errMsg) {
    llvm::SmallVector<char> buf;
    llvm::Error err = llvm::sys::fs::readNativeFileToEOF(
        llvm::sys::fs::convertFDToNativeFile(handle), buf);
    if (err) {
      *errMsg = copyString(llvm::toString(std::move(err)));
      return llvm::StringRef(nullptr, 0);
    }

    return copyString(llvm::StringRef(buf.data(), buf.size()));
  }

  llvm::StringRef readBytes(llvm::StringRef *errMsg) {
    llvm::SmallVector<char> buf;
    llvm::Error err = llvm::sys::fs::readNativeFileToEOF(
        llvm::sys::fs::convertFDToNativeFile(handle), buf);
    if (err) {
      *errMsg = copyString(llvm::toString(std::move(err)));
      return {nullptr, 0};
    }

    return copyBytes(buf);
  }

  uint64_t seek(uint64_t offset, llvm::StringRef *errMsg) {
#ifdef _WIN32
    return _lseeki64(handle, offset, SEEK_SET);
#else
    return lseek(handle, offset, SEEK_SET);
#endif
  }

  void write(llvm::StringRef buf, llvm::StringRef *errMsg) {
#ifdef _WIN32
    llvm::raw_fd_ostream os(handle, /*shouldClose=*/false, /*unbuffered=*/true);
#else
    llvm::raw_fd_ostream os(llvm::sys::fs::convertFDToNativeFile(handle),
                            /*shouldClose=*/false, /*unbuffered=*/true);
#endif // _WIN32
    os << buf;
  }

  void close(llvm::StringRef *errMsg) {
    llvm::sys::fs::file_t file = llvm::sys::fs::convertFDToNativeFile(handle);
    if (std::error_code err = llvm::sys::fs::closeFile(file))
      *errMsg = copyString(err.message());
  }

private:
  FileHandle(int handle) : handle(handle) {}
  FileHandle(const FileHandle &other) = delete;
  void operator=(const FileHandle &other) = delete;

  static llvm::sys::fs::FileAccess getFileAccess(llvm::StringRef mode) {
    llvm::sys::fs::FileAccess res = (llvm::sys::fs::FileAccess)0;

    if (mode.contains("r"))
      res |= llvm::sys::fs::FileAccess::FA_Read;

    if (mode.contains("w"))
      res |= llvm::sys::fs::FileAccess::FA_Write;

    return res;
  }

  int handle = 0;
};

struct FileHandleWrapper {
  void *ptr;
};

} // namespace

static FileHandle *unwrap(FileHandleWrapper ref) {
  return reinterpret_cast<FileHandle *>(ref.ptr);
}

static FileHandleWrapper wrap(FileHandle *ptr) {
  return FileHandleWrapper{ptr};
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT FileHandleWrapper
KGEN_CompilerRT_IO_FileOpen(llvm::StringRef path, llvm::StringRef mode,
                            llvm::StringRef *errMsg) {
  return wrap(FileHandle::get(path, mode, errMsg));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_IO_FileClose(FileHandleWrapper file, llvm::StringRef *errMsg) {
  FileHandle *handle = unwrap(file);
  handle->close(errMsg);
  delete handle;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_IO_FileSize(FileHandleWrapper file, llvm::StringRef *errMsg) {
  return unwrap(file)->size(errMsg);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT uint64_t
KGEN_CompilerRT_IO_FileSeek(FileHandleWrapper file, uint64_t offset,
                            llvm::StringRef *errMsg) {
  return unwrap(file)->seek(offset, errMsg);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_IO_FileRead(FileHandleWrapper file, uint64_t *size,
                            llvm::StringRef *errMsg) {
  llvm::StringRef str = unwrap(file)->read(errMsg);
  *size = str.size();
  return str.data();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT const char *
KGEN_CompilerRT_IO_FileReadBytes(FileHandleWrapper file, uint64_t *size,
                                 llvm::StringRef *errMsg) {
  llvm::StringRef str = unwrap(file)->readBytes(errMsg);
  *size = str.size();
  return str.data();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_IO_FileWrite(FileHandleWrapper file, const char *data,
                             uint64_t size, llvm::StringRef *errMsg) {
  unwrap(file)->write(llvm::StringRef(data, size), errMsg);
}

void M::KGEN::registerIO(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileOpen", (void *)&KGEN_CompilerRT_IO_FileOpen});
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileClose", (void *)&KGEN_CompilerRT_IO_FileClose});
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileSize", (void *)&KGEN_CompilerRT_IO_FileSize});
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileSeek", (void *)&KGEN_CompilerRT_IO_FileSeek});
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileRead", (void *)&KGEN_CompilerRT_IO_FileRead});
  funcs.push_back(
      {"KGEN_CompilerRT_IO_FileWrite", (void *)&KGEN_CompilerRT_IO_FileWrite});
  funcs.push_back({"KGEN_CompilerRT_IO_FileReadBytes",
                   (void *)&KGEN_CompilerRT_IO_FileReadBytes});
}
