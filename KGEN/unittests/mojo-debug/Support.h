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
  MojoBinary(const MojoSource &source, bool suppressBuildOutput = false);

  StringRef getPath() const { return binPath; }

private:
  MojoSource source;
  TempDir outDir;
  std::string binPath;
};

struct StopContext {
  MojoBinary binary;
  lldb::SBTarget target;
  lldb::SBProcess process;
  lldb::SBThread thread;
  lldb::SBFrame frame;

  /// Step over the current thread and return the StopContext once it stops.
  StopContext stepOver();
};

/// Builds the given source file, then creates a target with the resultant
/// binary, places breakpoints on all the locations with the `# breakpoint`
/// comment, and yields at the first stop.
StopContext buildAndLaunch(StringRef fileName);

} // namespace M

#endif // KGEN_UNITTESTS_MOJO_DEBUG_SUPPORT_H
