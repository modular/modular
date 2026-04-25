//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/DocGen.h"

#include "Config/Version.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Driver/DiagnosticFormat.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;

bool M::generateMojoDocJSON(const std::filesystem::path &resolvedPath,
                            mlir::MLIRContext &context,
                            const DocGenConfig &config, llvm::raw_ostream &os) {
  llvm::SourceMgr sourceManager;
  sourceManager.setDiagHandler(getDiagHandler(config.diagnosticFormat));
  sourceManager.setIncludeDirs(config.includePaths);

  CompilationOptions compilationOptions;
  compilationOptions.warningsAsErrors = config.warningsAsErrors;
  LIT::ParserConfig parserConfig(&context, compilationOptions);
  parserConfig.diagnoseMissingDocStrings = config.diagnoseMissingDocStrings;
  parserConfig.maxNotesPerDiagnostic = config.maxNotesPerDiagnostic;
  parserConfig.stripFilePrefix = config.stripFilePrefix;
  parserConfig.docsBasePath = config.docsBasePath;

  MojoParserContext parserContext(sourceManager, parserConfig);
  MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(resolvedPath);
  if (!moduleDecl || parserContext.wasErrorEmitted())
    return false;

  std::unique_ptr<PublicDecl> publicDecl = moduleDecl.getDecl();
  if (!publicDecl)
    return false;

  llvm::json::OStream jsonOS(os, /*IndentSize=*/2);
  const char *version = getMojoVersionString();
  jsonOS.value(llvm::json::Object({
      {"decl", publicDecl->toJSON(parserContext)},
      {"version", llvm::formatv("{0}", version).str()},
  }));

  return true;
}
