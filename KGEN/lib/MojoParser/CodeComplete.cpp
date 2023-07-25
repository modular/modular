//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CodeComplete.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

using llvm::SMLoc;

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a listener that collects code completion results.
struct CodeCompletionListener : public MojoParserListener {
  CodeCompletionListener(std::vector<CodeCompletionResult> &results, SMLoc loc,
                         llvm::SourceMgr &sourceMgr)
      : results(results), loc(loc), sourceMgr(sourceMgr) {}
  ~CodeCompletionListener() override = default;

  /// Notify the listener that an import is currently being resolved.
  void onImport(SMLoc importLoc) override {
    if (loc != importLoc)
      return;

    // Simple helper for adding completion results and dropping duplicates.
    StringSet<> addedImports;
    auto addImportCompletion = [&](StringRef name, bool isPackage) {
      if (addedImports.insert(name).second)
        results.emplace_back(name, isPackage ? CodeCompletionResult::kPackage
                                             : CodeCompletionResult::kModule);
    };

    // Compute the viable imports for the given location.
    for (const std::string &dir :
         parserContext->getModuleSearchDirectories(sourceMgr.getMainFileID())) {
      std::error_code ec;
      for (const auto &it : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
          continue;
        std::string extension = it.path().extension().string();
        if (extension == ".mojo" || extension == ".🔥")
          addImportCompletion(it.path().stem().string(), /*isPackage=*/false);
        else if (extension == ".mojopkg" || extension == ".📦" ||
                 isMojoSourcePackagePath(it.path()))
          addImportCompletion(it.path().stem().string(), /*isPackage=*/true);
      }
    }
  }

  /// The results that have been collected so far.
  std::vector<CodeCompletionResult> &results;

  /// The location of the code completion request.
  SMLoc loc;

  /// The source manager.
  llvm::SourceMgr &sourceMgr;

  /// The current parser context.
  MojoParserContext *parserContext = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

std::vector<CodeCompletionResult>
Mojo::codeComplete(llvm::MemoryBufferRef buffer, uint64_t completionPosition,
                   MLIRContext *context, LLCL::Runtime &runtime,
                   const KGEN::CompilationOptions &options) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(buffer),
                               SMLoc());

  // Add a diagnostic handler that consumes anything emitted during parsing. We
  // don't care about diagnostics here, there will almost always be a diagnostic
  // emitted when grabbing completion results from a partial file.
  sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});

  // Compute the completion SM location by finding the next token from the input
  // completion position.
  SMLoc completeLoc = SMLoc::getFromPointer(
      buffer.getBuffer().drop_front(completionPosition).ltrim().data());

  // Build the listener that collects the results.
  std::vector<CodeCompletionResult> results;
  CodeCompletionListener listener(results, completeLoc, sourceMgr);

  MojoParserConfig config(context, runtime, options);
  config.parserListener = &listener;

  // We don't want to cache the main module, but imported modules can be cached.
  config.moduleCachingLevel = MojoParserConfig::kCacheImports;

  // Disable as much of the diagnostic machinery as possible, we don't care
  // about diagnostics for completion results.
  config.maxNotesPerDiagnostic = 0;

  MojoParserContext parserContext(sourceMgr, config);
  listener.parserContext = &parserContext;
  parserContext.parseFile(sourceMgr.getMainFileID());

  return results;
}
