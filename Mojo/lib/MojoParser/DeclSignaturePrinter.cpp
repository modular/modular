//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Renders Mojo-syntax signatures directly from MLIR ops. This is the canonical
// implementation used by both the compiler diagnostic path and the mojo-doc
// tool.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/MojoDiags.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Diagnostic-oriented helpers
//===----------------------------------------------------------------------===//

std::string M::KGEN::synthesizeDeclSignature(const ASTDecl &decl) {
  std::string out;
  llvm::raw_string_ostream os(out);
  if (auto *fnDecl = dyn_cast<DefASTDecl>(&decl))
    printFunctionSignature(const_cast<DefASTDecl &>(*fnDecl), os);
  else if (auto *structDecl = dyn_cast<StructASTDecl>(&decl))
    printStructSignature(const_cast<StructASTDecl &>(*structDecl), os);
  else if (auto *aliasDecl = dyn_cast<AliasASTDecl>(&decl))
    printAliasSignature(const_cast<AliasASTDecl &>(*aliasDecl), os);
  else if (auto *traitDecl = dyn_cast<TraitASTDecl>(&decl))
    printTraitSignature(const_cast<TraitASTDecl &>(*traitDecl), os);
  return out;
}

bool M::KGEN::hasReadableSourceLocation(Location loc, SharedState &shared) {
  auto &diags = shared.diags;
  auto &sourceMgr = diags.sourceMgr;
  return sourceMgr.FindBufferContainingLoc(diags.convertLocToSMLoc(loc)) != 0;
}

MojoInflightDiag &MojoInflightDiag::attachNote(const ASTDecl &ctxDecl) & {
  auto *shared = getSharedIfActive();
  if (!shared)
    return *this;

  Operation *op = ctxDecl.getIfOperation();
  // Prefer the operation's location if available; it is often more correct in
  // the case of synthetic functions.
  Location loc =
      op ? op->getLoc() : getDiags()->translateLocation(ctxDecl.getLoc());

  auto &note = attachNote(loc);

  // Print synthetic functions differently, mentioning that they're generated
  // functions. If we can, prefer to print with a synthesized decl signature.
  // If we can't, print out the ASTType directly.
  if (auto fnOp = dyn_cast_if_present<FnOp>(op); fnOp && fnOp.isSynthetic()) {
    if (shared) {
      note.addCustomLineText(synthesizeDeclSignature(ctxDecl));
    } else {
      std::string out;
      llvm::raw_string_ostream os(out);
      os << ASTType(fnOp.getFullSignature());
      note.addCustomLineText(out);
    }
    note.addCustomLineText("    # note - generated function");
    return note;
  }

  // If the location is readable, just defer to adding a regular note
  if (hasReadableSourceLocation(loc, *shared))
    return note;

  // Else synthetize a signature for this op and print that
  if (auto sig = synthesizeDeclSignature(ctxDecl); !sig.empty()) {
    note.addCustomLineText(sig);
    note.addCustomLineText("    # note - synthetic signature");
  }

  return note;
}

MojoInflightDiag &MojoInflightDiag::attachNote(Location loc, TypedAttr attr) & {
  auto *shared = getSharedIfActive();
  if (!shared)
    return *this;

  auto &note = attachNote(loc);

  if (hasReadableSourceLocation(loc, *shared))
    return note;

  std::string out;
  llvm::raw_string_ostream os(out);
  ASTType::printParam(os, attr, /*ctx=*/{shared});

  if (!out.empty())
    note.addCustomLineText(out);

  return note;
}
