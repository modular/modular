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
//
// Tests RMO::materializePythonFrameLocations, which converts OpaqueLocs that
// wrap a PythonFrameAttr into serializable FusedLocs carrying the stack-trace
// string. OpaqueLoc cannot be written in textual MLIR (it holds a live frame
// pointer), so this behavior is covered here rather than with a lit test.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace M;

namespace {

// Returns a heap copy of a fixed trace string. PythonFrameAttr's storage owns
// and deletes the result, mirroring the real frame_traceback callback.
char *fakeTrace(void *) {
  const char *trace = "  File \"model.py\", line 42, in build\n";
  char *out = new char[std::strlen(trace) + 1];
  std::strcpy(out, trace);
  return out;
}

void noopRef(void *) {}

mlir::OpaqueLoc makePythonFrameLoc(mlir::MLIRContext *ctx) {
  RMO::PythonFrame frame{};
  frame.ptr = ctx; // any non-null pointer; refcount callbacks are no-ops.
  frame.to_str = &fakeTrace;
  frame.incref = &noopRef;
  frame.decref = &noopRef;
  return RMO::PythonFrameAttr::getOpaque(ctx, frame);
}

// An OpaqueLoc wrapping a PythonFrameAttr becomes a FusedLoc whose metadata is
// the stack-trace string, with the opaque loc's fallback preserved.
TEST(materializePythonFrameLocations, convertsOpaqueLocToFusedLoc) {
  mlir::MLIRContext ctx{mlir::MLIRContext::Threading::DISABLED};
  ctx.loadDialect<RMO::RMODialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(makePythonFrameLoc(&ctx)));

  RMO::materializePythonFrameLocations(module.get().getOperation());

  auto fused = mlir::dyn_cast<mlir::FusedLoc>(module.get().getLoc());
  ASSERT_TRUE(fused) << "expected a FusedLoc, got " << module.get().getLoc();
  auto trace = mlir::dyn_cast_or_null<mlir::StringAttr>(fused.getMetadata());
  ASSERT_TRUE(trace);
  EXPECT_EQ(trace.getValue(), "  File \"model.py\", line 42, in build\n");
}

} // namespace
