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

#ifndef GLIDIR_VIEWS_TEXTVIEW_H
#define GLIDIR_VIEWS_TEXTVIEW_H

#include "Common.h"
#include "TextViewer/TextViewer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

// Lets other views drive the IR text panel, which owns the op-to-line mapping.
struct TextViewCallbacks {
  std::function<void(int)> highlightOpInTextViewer;
  // Also expands the op in the tree view.
  std::function<void(mlir::Operation *)> navigateToOp;
  // Leaves the tree view alone.
  std::function<void(mlir::Operation *)> scrollToOpInTextViewer;
};

// Read-only, syntax-highlighted IR dump, with a line<->op mapping that keeps
// it in sync with the other panels.
class TextView {
public:
  // Fired on cursor moves and bookmark toggles to sync the tree view.
  struct TreeCallbacks {
    // nullptr if there's no op on the cursor's line.
    std::function<void(mlir::Operation *)> highlightOp;
    std::function<void()> clearHighlight;
  };

  TextView() = default;

  void setTreeCallbacks(TreeCallbacks cbs) { treeCallbacks = cbs; }

  // Call once before rendering. dumpRoot is root re-parsed with locations
  // into dumpText; both are walked in parallel to map lines to ops.
  void init(mlir::ModuleOp root, mlir::ModuleOp dumpRoot,
            const std::string &dumpText, const TextViewer::Palette &palette);

  TextViewer &getViewer() { return textViewer; }

  const llvm::DenseMap<mlir::Operation *, int> &getOpLineNumber() const {
    return opLineNumber;
  }

  // Lines without an op (e.g. region / block headers) map to nullptr.
  const llvm::SmallVector<mlir::Operation *> &getLineNumberToOp() const {
    return lineNumberToOp;
  }

  // The color group for new double-click bookmarks.
  int getActiveBookmarkColor() const { return activeBookmarkColor; }
  void setActiveBookmarkColor(int idx) { activeBookmarkColor = idx; }
  bool hasAnyBookmarks() const;
  // 1-indexed, ascending.
  llvm::SmallVector<int> sortedBookmarksForGroup(int idx) const;
  // In line order.
  llvm::SmallVector<mlir::Operation *> opsForGroup(int idx) const;
  // Lines are 0-indexed and inclusive.
  void bookmarkSelection(int minLine, int maxLine, int colorIdx);

  // lineNo is 0-indexed; notifies the tree view.
  void highlightLine(int lineNo);

  void scrollToOp(mlir::Operation *op);

  void renderTextFilter();

  // Fires tree callbacks on cursor moves; double-click toggles a bookmark.
  void processTextViewerCursorEvents();

  void renderBookmarks();

private:
  TextViewer textViewer;
  // For regex search.
  std::vector<std::string> textAsLines;
  llvm::SmallVector<mlir::Operation *> lineNumberToOp;
  llvm::DenseMap<mlir::Operation *, int> opLineNumber;
  llvm::SmallVector<int> matchedLines;
  std::optional<int> currentMatchedIndex;
  TreeCallbacks treeCallbacks;
  // 1-indexed line numbers.
  std::array<std::set<int>, kNumBookmarkColors> coloredBookmarks;
  int activeBookmarkColor = 0;

  // Breakpoints are the union of all coloredBookmarks.
  void syncBreakpoints();
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_TEXTVIEW_H
