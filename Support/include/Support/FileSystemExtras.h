//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FILESYSTEM_EXTRAS_H
#define SUPPORT_FILESYSTEM_EXTRAS_H

#include "LLVMForwardDecls.h"
#include "Support/ForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <filesystem>

namespace M {

/// This function searches for an existing subdirectory in the list of
/// directories in a PATH like environment variable, and returns the first
/// subdirectory found according to the order of the entries in the PATH like
/// environment variable.
std::optional<std::string> findDirInEnvPath(StringRef subdirName,
                                            StringRef envName = "PATH",
                                            char separator = ':');

/// Safely process creating and writing the file, taking into account that we
/// may have different processes trying to produce this file in parallel.
ErrorOr<std::filesystem::path>
writeFileAtomically(const std::filesystem::path &filePath,
                    llvm::function_ref<void(raw_ostream &)> writeContent);

/// This class provides a tempfile implementation. The llvm::sys version has
/// some really odd behavior that is tricky to manage, so we provide our own
/// implementation.
class TempFile {
public:
  /// Create a TempFile and return any errors during creation. The model is
  /// something like `myString-%%%%%.ext` - the `%` characters are filled in
  /// with random numbers/letters.
  static ErrorOr<TempFile> create(StringRef model);
  /// TempFiles are move-able.
  TempFile(TempFile &&other);
  /// Destroy the temp file, and remove it from the filesystem if `keepFile` is
  /// not specified.
  ~TempFile();

  /// Keep the tempfile after the destructor runs - useful for debugging.
  void keep() { keepFile = true; }

  /// Get the file descriptor as an integer. This file is open as of the
  /// completion of the `create` call.
  int getFD() { return fd; }
  /// Return the path to the temp file. This path is absolute.
  const std::filesystem::path &getPath() const { return path; }
  /// Get the size of the temp file in bytes.
  ErrorOr<size_t> getSize();

private:
  TempFile(int fd, std::string path) : fd(fd), path(std::move(path)) {}
  /// These are not copy-able.
  TempFile(const TempFile &other) = delete;

  int fd = -1;
  std::filesystem::path path;
  bool keepFile = false;
};
} // namespace M

#endif // SUPPORT_FILESYSTEM_EXTRAS_H
