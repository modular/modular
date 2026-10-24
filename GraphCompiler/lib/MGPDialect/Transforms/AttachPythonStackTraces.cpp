//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/RMODialect/RMOAttrs.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "mlir/IR/Visitors.h"

#define DEBUG_TYPE "attach-python-stack-traces"

namespace M::MGP {

#define GEN_PASS_DEF_ATTACHPYTHONSTACKTRACES
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

}; // namespace M::MGP

using namespace M;
using namespace MGP;

namespace {

static llvm::StringRef extractStackTrace(mlir::OpaqueLoc opaqueLoc) {
  if (auto pythonFrame = RMO::PythonFrameAttr::fromOpaque(opaqueLoc))
    return pythonFrame.stack_trace();
  return {};
}

//===----------------------------------------------------------------------===//
// AttachPythonStackTraces
//===----------------------------------------------------------------------===//
//
// For each op in the module that has an OpaqueLoc wrapping a PythonFrameAttr:
//  - Embeds the Python stack trace into the message of any mgp.assert op.
//  - Encodes the trace as FusedLoc metadata so downstream lowering can
//    retrieve it via collectSourceNotes (GEX-2678).
//  - Replaces each OpaqueLoc, including ones nested in a FusedLoc, with its
//    fallback; OpaqueLoc does not serialize to bytecode (GEX-1435).

struct AttachPythonStackTraces
    : public impl::AttachPythonStackTracesBase<AttachPythonStackTraces> {
  using AttachPythonStackTracesBase::AttachPythonStackTracesBase;

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();

    getOperation()->walk([&ctx](Operation *op) {
      Location loc = op->getLoc();

      // Only the op's immediate OpaqueLoc applies, not ones nested in a
      // FusedLoc.
      if (auto assertOp = dyn_cast<AssertOp>(op)) {
        if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {
          llvm::StringRef stackTrace = extractStackTrace(opaqueLoc);
          if (!stackTrace.empty()) {
            std::string fullMsg = "\nSource Traceback:\n";
            llvm::raw_string_ostream ss(fullMsg);
            ss << stackTrace << '\n';
            StringAttr msg = assertOp.getMessageAttr();
            mlir::Diagnostic diag(loc, mlir::DiagnosticSeverity::Error);
            diag << msg.str();
            FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(ss,
                                                                          diag);
            assertOp.setMessageAttr(StringAttr::get(ctx, fullMsg));
          }
        }
      }
    });

    RMO::materializePythonFrameLocations(getOperation());
  }
};

} // namespace
