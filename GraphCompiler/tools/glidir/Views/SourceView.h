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

// This view was vibe-coded by Claude.
//
// Displays source files referenced by FileLineColLoc locations in the IR.
// Uses a TextViewer so the source panel looks identical to the IR panel.

#ifndef GLIDIR_VIEWS_SOURCEVIEW_H
#define GLIDIR_VIEWS_SOURCEVIEW_H

#include "TextViewer/TextViewer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

// Collects FileLineColLoc leaves, including inside FusedLocs.
void collectFileLineColLocs(
    mlir::Location loc, llvm::SmallVectorImpl<mlir::FileLineColLoc> &result);

struct SourceView {
  // One entry per unique source file referenced by any op's FileLineColLoc.
  struct SourceFile {
    // Absolute; the tab label.
    std::string path;
    std::string content;
    llvm::DenseMap<int, llvm::SmallVector<mlir::Operation *>> lineToOp;
  };

  std::vector<SourceFile> files;
  int currentFileIndex = 0;
  // File loaded into textViewer; differs from currentFileIndex on tab switch.
  std::optional<int> loadedFileIndex;
  TextViewer textViewer;
  // Detects highlight changes so the viewer can scroll.
  mlir::Operation *prevHighlightedOp = nullptr;

  // ImGui input widgets need persistent buffers.
  char searchBuffer[512] = {};
  char prevSearchBuffer[512] = {};
  // Re-focus the search input so repeated Enter keeps navigating.
  bool searchRefocus = false;
  llvm::SmallVector<int> searchMatchedLines;
  std::optional<int> searchCurrentIndex;
  std::vector<std::string> cachedTextLines;
  // If non-negative, scroll here after loading a file.
  int pendingSearchLine = -1;

  SourceView() = default;
  // Loads every source file referenced by root's locations.
  explicit SourceView(mlir::ModuleOp root);

  // Call once the IR panel's palette is final (end of Renderer ctor).
  void initViewer(const TextViewer::Palette &palette);

  void renderSourceFilter();

  // onSourceLineClicked gets all ops mapped to the clicked line.
  void render(mlir::Operation *highlightedOp,
              std::function<void(llvm::ArrayRef<mlir::Operation *>)>
                  onSourceLineClicked);
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_SOURCEVIEW_H
