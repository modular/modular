//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DIBuilder.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// DIBuilder
//===----------------------------------------------------------------------===//

DICompileUnitAttr DIBuilder::initializeCompileUnit(
    unsigned sourceLanguage, DIFileAttr file, StringRef producer,
    bool isOptimized, EmissionKind emissionKind, NameTableKind nameTableKind) {
  assert(!compileUnit && "compile unit already initialized");
  compileUnit = DICompileUnitAttr::get(
      sourceLanguage, file, producer, isOptimized, emissionKind, nameTableKind);
  return compileUnit;
}

//===----------------------------------------------------------------------===//
// Scopes

void DIBuilder::pushScope(DIScopeAttr scope) { scopes.push_back(scope); }

void DIBuilder::popScope() {
  assert(!scopes.empty() && "Cannot pop the compile unit scope!");
  scopes.pop_back();
}

Location DIBuilder::createScopedLoc(Location loc) {
  if (scopes.empty() || !scopes.back())
    return loc;

  // Check if this is already a scoped location with the expected scope.
  if (auto scopedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc))
    if (scopedLoc.getMetadata() == scopes.back())
      return loc;
  return FusedLoc::get(loc.getContext(), loc, scopes.back());
}

//===----------------------------------------------------------------------===//
// Creation

DILexicalBlockAttr DIBuilder::createNestedLexicalBlock(DIFileAttr file,
                                                       unsigned line,
                                                       unsigned column) {
  if (!scopes.back())
    return nullptr;
  auto scope = cast<DILocalScopeAttr>(scopes.back());
  return DILexicalBlockAttr::get(scope, file, line, column);
}

DISubprogramAttr DIBuilder::createSubprogram(SourceNameAttr name,
                                             StringAttr linkageName,
                                             DIFileAttr file, unsigned int line,
                                             unsigned int scopeLine,
                                             SubprogramFlags subprogramFlags,
                                             DISubroutineType type) {
  // Get the last non-local scope to use as the parent for the subprogram.
  auto range = llvm::reverse(scopes);
  auto it = llvm::find_if(range, [](DIScopeAttr scope) {
    return !isa_and_nonnull<DILocalScopeAttr>(scope);
  });
  assert(it != range.end() &&
         "didn't find a non-local scope -- forgot to push one?");
  return DISubprogramAttr::get(compileUnit, *it, name, linkageName, file, line,
                               scopeLine, subprogramFlags, type);
}

DIFileAttr DIBuilder::createFile(StringRef name, StringRef directory) {
  return DIFileAttr::get(context, name, directory);
}
DIFileAttr DIBuilder::createFile(FileLineColLoc loc) {
  return DIFileAttr::get(context, loc.getFilename(), "/");
}

DILocalVariableAttr DIBuilder::createLocalVariable(StringRef name,
                                                   DIFileAttr file,
                                                   unsigned line, unsigned arg,
                                                   unsigned alignInBits,
                                                   DIType type) {
  auto scope = cast<DILocalScopeAttr>(scopes.back());
  return DILocalVariableAttr::get(scope, name, file, line, arg, alignInBits,
                                  type);
}
