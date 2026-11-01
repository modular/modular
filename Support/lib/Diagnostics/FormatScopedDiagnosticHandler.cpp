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

#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <string>
#include <utility>

using namespace M;
using namespace mlir;

static StringRef severityToString(mlir::DiagnosticSeverity severity) {
  if (severity == mlir::DiagnosticSeverity::Note)
    return "note";
  if (severity == mlir::DiagnosticSeverity::Warning)
    return "warning";
  if (severity == mlir::DiagnosticSeverity::Error)
    return "error";
  if (severity == mlir::DiagnosticSeverity::Remark)
    return "remark";

  llvm_unreachable("Unexpected diagnostic severity enum value");
}

static std::string locationToString(Location location) {
  auto fileLoc = dyn_cast<mlir::FileLineColLoc>(location);
  if (!fileLoc)
    return std::string();

  return Twine(Twine(fileLoc.getFilename()) + ":" + Twine(fileLoc.getLine()) +
               ":" + Twine(fileLoc.getColumn()))
      .str();
}

namespace {
/// Prints one diagnostic directly, without going through the context's engine.
struct SourcePrinter : mlir::SourceMgrDiagnosticHandler {
  using SourceMgrDiagnosticHandler::emitDiagnostic;
  using SourceMgrDiagnosticHandler::SourceMgrDiagnosticHandler;
};
} // namespace

std::string
FormatScopedDiagnosticHandler::formatWithSource(const mlir::Diagnostic &diag) {
  // Drop the "see current operation" note: it dumps the op's IR, which says
  // nothing to someone reading their Mojo source.
  mlir::Diagnostic shown(diag.getLocation(), diag.getSeverity());
  shown << diag.str();
  for (const mlir::Diagnostic &note : diag.getNotes())
    if (!llvm::StringRef(note.str()).starts_with("see current operation"))
      shown.attachNote(note.getLocation()) << note.str();

  std::string out;
  llvm::raw_string_ostream os(out);
  // Without a file location there is no source line to show, and
  // `<unknown>:0:` says nothing.
  if (!diag.getLocation()->findInstanceOf<mlir::FileLineColLoc>()) {
    emitDiagnosticSeverityToStream(os, shown);
    for (const mlir::Diagnostic &note : shown.getNotes()) {
      os << '\n';
      emitDiagnosticSeverityToStream(os, note);
    }
    return out;
  }
  // The printer registers itself as a handler on its context while it lives;
  // a private context keeps that off the caller's, which may be emitting from
  // other threads or from inside a handler.
  mlir::MLIRContext printerContext(mlir::MLIRContext::Threading::DISABLED);
  // Loads each file the locations name on demand, as `mojo build` does.
  llvm::SourceMgr sourceMgr;
  SourcePrinter printer(sourceMgr, &printerContext, os);
  printer.emitDiagnostic(shown);
  return llvm::StringRef(out).rtrim().str();
}

void FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(
    raw_ostream &os, const Diagnostic &diag) {
  os << severityToString(diag.getSeverity()) << ": ";
  os << diag;
  os << "\n";
}

static void emitLocation(raw_ostream &minimalOutputStream,
                         raw_ostream &fullOutputStream,
                         const Diagnostic &diag) {
  // Only display the location if it is meaningful.
  std::string location = locationToString(diag.getLocation());
  if (!location.empty()) {
    minimalOutputStream << location << ": ";
    fullOutputStream << location << ": ";
  }
}

static void emitSeverity(raw_ostream &minimalOutputStream,
                         raw_ostream &fullOutputStream,
                         const Diagnostic &diag) {
  minimalOutputStream << severityToString(diag.getSeverity()) << ": ";
  fullOutputStream << severityToString(diag.getSeverity()) << ": ";
}

static void emitDiagnostic(raw_ostream &minimalOutputStream,
                           raw_ostream &fullOutputStream,
                           const Diagnostic &diag) {
  // First get the entire diagnostic as a string, so that we can inspect it.
  std::string diagString;
  llvm::raw_string_ostream diagStream(diagString);
  diagStream << diag;
  diagStream.flush();

  size_t newlinePos = diagString.find('\n');
  if (newlinePos != std::string::npos) {
    // There is a new line, so emit the first line only to the minimal stream.
    minimalOutputStream << diagString.substr(0, newlinePos);
    if (newlinePos + 1 < diagString.size()) {
      // There is more to the message that we didn't emit, add a note to the
      // minimal stream to indicate that the message was elided.
      minimalOutputStream << " (additional lines ("
                          << (diagString.size() - newlinePos - 1)
                          << " bytes) elided)";
    }
  } else {
    // There is no new line, emit the entire message to the minimal stream.
    minimalOutputStream << diagString;
  }
  minimalOutputStream << "\n";

  // Always emit the entire message to the full stream.
  fullOutputStream << diagString << "\n";
}

static void emitIndentation(raw_ostream &minimalOutputStream,
                            raw_ostream &fullOutputStream,
                            const Diagnostic &diag) {
  const char *indentation = "  ";
  minimalOutputStream << indentation;
  fullOutputStream << indentation;
}

void FormatScopedDiagnosticHandler::emitDiagnosticToStream(
    raw_ostream &fullOutputStream, const Diagnostic &diag) {
  llvm::raw_null_ostream nullStream;
  emitDiagnosticToStream(nullStream, fullOutputStream, diag);
}

void FormatScopedDiagnosticHandler::emitDiagnosticToStream(
    raw_ostream &minimalOutputStream, raw_ostream &fullOutputStream,
    const Diagnostic &diag) {

  emitLocation(minimalOutputStream, fullOutputStream, diag);
  emitSeverity(minimalOutputStream, fullOutputStream, diag);
  emitDiagnostic(minimalOutputStream, fullOutputStream, diag);

  // Display each note, indented two spaces
  for (Diagnostic &note : diag.getNotes()) {
    emitIndentation(minimalOutputStream, fullOutputStream, note);
    emitDiagnosticToStream(minimalOutputStream, fullOutputStream, note);
  }
}

FormatScopedDiagnosticHandler::FormatScopedDiagnosticHandler(MLIRContext *ctx)
    : mlir::ScopedDiagnosticHandler(ctx, [&](Diagnostic &diag) {
        diagnostics.push_back(std::move(diag));
      }) {}

std::string FormatScopedDiagnosticHandler::formatMessage() const {
  std::string message;
  llvm::raw_string_ostream messageStream(message);
  for (auto &diagnostic : diagnostics)
    emitDiagnosticToStream(messageStream, diagnostic);
  return message;
}
