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
/// environment variable. The defaults hold for Unix-like systems but break for
/// Windows.
std::optional<std::string> findDirInEnvPath(StringRef subdirName,
                                            StringRef envName = "PATH",
                                            char separator = ':');

/// Write to a file (creating if necessary) serialized with any other
/// ...UnderLock operation, even in parallel across processes.  Writing will
/// also appear atomic to readers not aware of LLVM lock files.
ErrorOr<std::filesystem::path>
writeFileUnderLock(const std::filesystem::path &filePath,
                   llvm::function_ref<void(raw_ostream &)> writeContent);

/// Read a file exclusively, serializing with other ...UnderLock operations,
/// even in parallel across processes.  Other processes using readFileUnderLock
/// will wait for the operation to complete before initating their reads.  The
/// operation is only atomic with respect to processes abiding by the LLVM lock
/// file convention -- no atomicity guarantees are provided with respect to
/// writers not aware of the LLVM lock file convention concurrently operating
/// on the file.
ErrorOrSuccess
readFileUnderLock(const std::filesystem::path &filePath,
                  llvm::function_ref<void(const std::filesystem::path &)> read);

/// Append to a file exclusively, serializing with other ...UnderLock
/// operations, even in parallel across processes.  Other processes appending
/// will block while the append is in progress.  If the process crashes in the
/// middle of appending, other processes may witness a partially-appended
/// state.  Processes not aware of the LLVM lock file convention may also
/// witness partially-appended states while the append is in progress.
ErrorOrSuccess
appendFileUnderLock(const std::filesystem::path &filePath,
                    llvm::function_ref<void(raw_ostream &)> appendContent);

/// Invokes the provided callback, writing the output to a temporary file whose
/// name is based on the provided model. On success, `outPath` is populated with
/// the path of temporary file.
ErrorOrSuccess writeTempFile(const Twine &model,
                             function_ref<void(raw_ostream &)> writeFn,
                             std::string &outPath);
ErrorOrSuccess writeTempFile(const Twine &model, StringRef buffer,
                             std::string &outPath);

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
