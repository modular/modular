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

// This view was vibe-coded by Claude and is not yet well understood.
//
// Interactive use-def graph view with an iongraph-inspired Sugiyama-style
// layered layout (https://spidermonkey.dev/blog/2025/10/28/iongraph-web.html):
//   1. BFS layering (entry = layer 0, each layer flows downward)
//   2. Backedge detection via DFS (drawn as right-side loops)
//   3. Within-layer centering with a single parent-alignment pass
//   4. Railroad-style elbow edges (vertical + horizontal segments)
//   5. Infinite canvas with pan (drag) and zoom (Ctrl+scroll)

#ifndef GLIDIR_VIEWS_GRAPHVIEW_H
#define GLIDIR_VIEWS_GRAPHVIEW_H

#include "Common.h"
#include "TextView.h"

#include "imgui.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

struct GraphView {
  // One per bookmarked op.
  struct Node {
    mlir::Operation *op = nullptr;
    // Layer 0 is the topmost row.
    int layer = 0;
    // World (pre-zoom) coordinates.
    float x = 0, y = 0, w = 0, h = 0;
    // "dialect.opname" string shown in the coloured header stripe.
    std::string name;
    // Printed op line, truncated to 72 chars.
    std::string text;
    // `text` truncated to the node width; recomputed each frame.
    std::string displayText;
  };

  struct Edge {
    int src = 0, dst = 0;
    // Spread attachment points evenly across the node's bottom/top edge.
    int srcSlot = 0, srcCount = 1;
    int dstSlot = 0, dstCount = 1;
    // Hover tooltip.
    std::string valueText;
  };

  // Layout constants (world units).
  static constexpr float kPadX = 10, kPadY = 4;
  static constexpr float kLayerGap = 60, kNodeGap = 24;
  static constexpr float kMaxNodeW = 320.0f;

  llvm::SmallVector<Node, 0> nodes;
  llvm::SmallVector<Edge, 0> edges;
  // Rebuilt whenever bookmarks change.
  llvm::DenseMap<mlir::Operation *, int> opIndex;
  // Compared each frame to detect changes requiring a rebuild.
  llvm::DenseSet<mlir::Operation *> lastBookmarkSet;
  // Avoids firing navigateToOp every frame while hovering one node.
  mlir::Operation *lastHoveredOp = nullptr;

  TextViewCallbacks callbacks;
  // Screen pixels.
  ImVec2 pan = {300, 80};
  float zoom = 1.0f;
  int activeGroupIndex = 0;
  // Consumed by the renderer after the frame reaches the back buffer.
  bool pendingSave = false;
  // Captured during render(): buildExportContext() runs after ImGui::Render(),
  // when GetFontSize()/GetFont() are unreliable.
  float lastLH = 13.0f;
  ImFont *lastFont = nullptr;
  // Last canvas rect in ImGui logical coordinates, for cropping the capture.
  ImVec2 lastCanvasOrigin{}, lastCanvasEnd{};

  // Draw list and transform for the live canvas or an offscreen export.
  struct DrawContext {
    ImDrawList *dl;
    ImVec2 origin;
    ImVec2 canvasEnd;
    // origin + pan.
    float panX, panY;
    float zoom;
    float lh;
    // ImGui::GetFont() returns null after Render().
    ImFont *font = nullptr;

    ImVec2 w(float wx, float wy) const {
      return {origin.x + panX + wx * zoom, origin.y + panY + wy * zoom};
    }
  };

  GraphView() = default;
  GraphView(mlir::ModuleOp, TextViewCallbacks &cb);

  void buildGraph(llvm::ArrayRef<mlir::Operation *> ops);
  void computeLayout();
  // hoveredOp, selectedOp, and hoveredEdge may be null.
  void drawContent(const DrawContext &ctx, mlir::Operation *hoveredOp,
                   mlir::Operation *selectedOp, const Edge *hoveredEdge) const;
  // Fits every node into an offscreen image, setting outW/outH; scale=2
  // doubles both dimensions.
  DrawContext buildExportContext(ImDrawList *dl, int &outW, int &outH,
                                 float scale = 1.5f) const;
  // groups[i] holds color group i's ops. Rebuilds the graph when the active
  // group changes.
  void render(llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>> groups,
              mlir::Operation *selectedOp);
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_GRAPHVIEW_H
