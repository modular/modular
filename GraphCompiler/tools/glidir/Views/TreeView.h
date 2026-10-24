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

#ifndef GLIDIR_VIEWS_TREEVIEW_H
#define GLIDIR_VIEWS_TREEVIEW_H

#include "TextView.h"

#include "imgui.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace M {
namespace GlidIR {

struct TreeView {
  TextViewCallbacks callbacks;
  mlir::ModuleOp root;
  // Set by the text or graph panel on cursor move or hover.
  mlir::Operation *opToHighlight = nullptr;
  // Cleared once applied so the view doesn't snap back every frame.
  bool adjustYPosToHighlightedOp = true;
  // Ancestors to expand for opToHighlight, consumed in one frame.
  llvm::DenseSet<void *> mlirEntitiesToExpandForHighlightedOp;
  // For one frame after the highlight changes, force the path to root open;
  // afterwards the user's collapse state wins.
  bool pendingExpansion = false;
  // E.g. from a graph-panel double-click; independent of opToHighlight.
  mlir::Operation *opToFocus = nullptr;
  // Screen-space centres this frame, for renderUsers()'s arrows.
  llvm::DenseMap<void *, ImVec2> renderedMLIREntities;
  // User expand/collapse state across frames.
  llvm::DenseMap<void *, bool> mlirEntitiesPreviouslyExpanded;
  // Where "collapse children" was triggered; its children stay closed until
  // re-expanded.
  void *mlirEntityWhereWeCollapsed = nullptr;
  struct DetachedWindow {
    std::string title;
    std::function<void(bool &)> renderFn;
    llvm::DenseMap<void *, bool> expandedState;
    DetachedWindow(std::string title, std::function<void(bool &)> renderFn)
        : title(std::move(title)), renderFn(std::move(renderFn)) {}
  };
  std::vector<DetachedWindow> detachedWindows;
  // Unique ImGui IDs via the ###id suffix.
  int nextDetachedId = 0;

  // Calls ImGui::TreePop() on destruction if the node was open.
  struct CollapsibleTreeNodeGuard {
    CollapsibleTreeNodeGuard(bool recursiveCollapseBeforeThis, bool wasExpanded)
        : recursiveCollapseBeforeThis(recursiveCollapseBeforeThis),
          wasExpanded(wasExpanded) {}
    ~CollapsibleTreeNodeGuard() {
      if (wasExpanded || recursiveCollapseBeforeThis)
        ImGui::TreePop();
    }

    operator bool() const { return wasExpanded; }

    bool recursiveCollapseBeforeThis;
    bool wasExpanded;
  };

  TreeView() = default;
  TreeView(mlir::ModuleOp root, TextViewCallbacks &callbacks);

  void clearHighlightedOp();
  void highlightOpOnTreeView(mlir::Operation *op);
  void clearPerFrameData();
  void render();

  static void drawElbowArrow(ImDrawList *dl, ImVec2 src, ImVec2 dst,
                             ImU32 color);
  void renderUsers();
  void applyExpandState(void *entity);

  void render(mlir::Block &block, uint64_t id, bool recursiveCollapse);
  void render(mlir::Region &region, bool recursiveCollapse);
  void render(mlir::Type type, bool recursiveCollapse);

  template <typename T>
  static void setToolTipOnMLIREntity(T mlirEntity) {
    if (ImGui::BeginItemTooltip()) {
      std::string str;
      llvm::raw_string_ostream stream(str);
      mlirEntity.print(stream);
      ImGui::Text("%s", str.c_str());
      ImGui::EndTooltip();
    }
  }

  void render(mlir::Attribute attr, bool recursiveCollapse);

  // Pops the style colour before the context menu so its text isn't tinted.
  // openFn adds an "Open in new window" item.
  CollapsibleTreeNodeGuard
  ColouredCollapsibleNode(ImU32 colour, bool &recursiveCollapse,
                          const char *name, ImGuiTreeNodeFlags flags = 0,
                          void *mlirEntity = nullptr,
                          std::function<void()> openFn = nullptr);

  void render(mlir::DictionaryAttr dict, bool recursiveCollapse);

  template <typename Range>
  void renderValueRange(Range values, bool recursiveCollapse,
                        const char *groupIcon, const char *groupLabel,
                        const char *itemIcon) {
    if (values.empty())
      return;

    mlir::OpPrintingFlags flags;
    flags.useLocalScope(false);

    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    static constexpr ImU32 kColour = IM_COL32(0, 255, 0, 255);
    auto groupLabelStr = llvm::formatv("{}{}", groupIcon, groupLabel).str();
    auto treeNode = ColouredCollapsibleNode(kColour, recursiveCollapse,
                                            groupLabelStr.c_str());
    if (treeNode || recursiveCollapse) {
      for (auto &&value : values) {
        std::string valueStr;
        llvm::raw_string_ostream stream(valueStr);
        value.printAsOperand(stream, flags);

        if (recursiveCollapse)
          ImGui::SetNextItemOpen(false, ImGuiCond_Always);

        auto itemLabel = llvm::formatv("{}{}", itemIcon, valueStr).str();
        mlir::Type valType = value.getType();
        std::string openTitle =
            llvm::formatv("{}{}###detached", itemIcon, valueStr).str();
        auto openFn = [this, valType, t = std::move(openTitle)]() mutable {
          int wid = nextDetachedId++;
          detachedWindows.push_back(DetachedWindow{
              llvm::formatv("{}###detached_{}", t, wid).str(),
              [this, valType](bool &rc) { render(valType, rc); }});
        };
        auto node = ColouredCollapsibleNode(kColour, recursiveCollapse,
                                            itemLabel.c_str(), 0, nullptr,
                                            std::move(openFn));
        setToolTipOnMLIREntity(value.getType());
        if (node || recursiveCollapse)
          render(value.getType(), recursiveCollapse);
      }
    }
  }

  void render(mlir::OperandRange operands, bool recursiveCollapse);
  void render(mlir::ResultRange results, bool recursiveCollapse);
  void registerEntityAsRendered(void *entity);
  void render(mlir::Operation &op, uint64_t id, bool recursiveCollapse);

  // "collapse children" sets recursiveCollapse, closing every later node on
  // this render path, so the whole subtree collapses in one frame. openFn adds
  // an "Open in new window" item.
  CollapsibleTreeNodeGuard
  CollapsibleTreeNode(bool &recursiveCollapse, const char *name,
                      ImGuiTreeNodeFlags flags = 0, void *mlirEntity = nullptr,
                      std::function<void()> openFn = nullptr);

  void renderDetachedWindows();
};

} // namespace GlidIR
} // namespace M

#endif // GLIDIR_VIEWS_TREEVIEW_H
