//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_UNITTESTS_MOJO_DEBUG_SUPPORT_H
#define KGEN_UNITTESTS_MOJO_DEBUG_SUPPORT_H

#include "../tools/mojo-lsp-test-client/LSPBatchClient.h"
#include "lldb/API/LLDB.h"

namespace lsp = mlir::lsp;

namespace M {

/// Class that represents a source file.
class MojoSource {
public:
  MojoSource(StringRef fileName);

  MojoSource(const MojoSource &) = delete;
  MojoSource &operator=(const MojoSource &) = delete;

  const std::filesystem::path &getFilesystemPath() const { return path; }

  StringRef getPath() const { return pathStr; }

  /// Generate the 1-indexed line numbers at which the given text is found
  /// in the source file.
  std::vector<int> findLinesWithText(StringRef text) const;

private:
  std::filesystem::path path;
  std::string pathStr;
  std::string contents;
  SmallVector<StringRef> lines;
};

/// Class that represents binary that is the result of compiling the given
/// source.
class MojoBinary {
public:
  MojoBinary(const std::shared_ptr<MojoSource> &source,
             bool suppressBuildOutput = false);

  StringRef getPath() const { return binPath; }

  const MojoSource &getSource() const { return *source; }

private:
  std::shared_ptr<MojoSource> source;
  TempDir outDir;
  std::string binPath;
};

struct CommandResult {
  bool success;
  std::string output;
  std::string error;
};

struct StopContext {
  /// Step over the current thread and return the StopContext once it stops.
  StopContext stepOver();

  /// Step int the current thread and return the StopContext once it stops.
  StopContext stepInto();

  /// Resume the current process and return the StopContext once it stops.
  StopContext resume();

  /// Run the given command using the current frame as context.
  CommandResult runCommand(StringRef command);

  MojoBinary binary;
  lldb::SBTarget target;
  lldb::SBProcess process;
  lldb::SBThread thread;
  lldb::SBFrame frame;
};

/// Builds the given source file, then creates a target with the resultant
/// binary, places breakpoints on all the locations with the `# breakpoint`
/// comment, and yields at the first stop.
StopContext buildAndLaunch(StringRef fileName);

} // namespace M

#endif // KGEN_UNITTESTS_MOJO_DEBUG_SUPPORT_H
