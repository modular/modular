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

#include "TreeView.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace M {
namespace GlidIR {

static constexpr auto kIconOp = "\xe2\x9a\x99 ";         // U+2699
static constexpr auto kIconBlock = "\xf0\x9f\x93\x82 ";  // U+1F4C2
static constexpr auto kIconRegion = "\xf0\x9f\x93\xa6 "; // U+1F4E6
static constexpr auto kIconOperands = "\xe2\x86\x92 ";   // →  operands group
static constexpr auto kIconOperand = "\xe2\x86\xaa ";    // ↪  operand
static constexpr auto kIconResults = "\xe2\x86\x90 ";    // ←  results group
static constexpr auto kIconResult = "\xe2\x86\xa9 ";     // ↩  result
static constexpr auto kIconAttrDict = "\xe2\x89\xa1 ";   // ≡  attr-dict
static constexpr auto kIconAttr = "\xf0\x9f\xa7\xb6 ";   // U+1F9F6
static constexpr auto kIconType = "\xe2\x8a\xa4 ";       // ⊤  type

TreeView::TreeView(ModuleOp root, TextViewCallbacks &callbacks)
    : callbacks(callbacks), root(root) {}

void TreeView::clearHighlightedOp() { opToHighlight = nullptr; }

void TreeView::highlightOpOnTreeView(Operation *op) {
  if (!op)
    return;

  opToHighlight = op;
  opToFocus = opToHighlight;
  adjustYPosToHighlightedOp = true;
  pendingExpansion = true;
  mlirEntitiesToExpandForHighlightedOp.clear();
  do {
    auto block = op->getBlock();
    auto region = op->getParentRegion();
    mlirEntitiesToExpandForHighlightedOp.insert(block);
    mlirEntitiesToExpandForHighlightedOp.insert(region);
    op = op->getParentOp();
  } while (op && mlirEntitiesToExpandForHighlightedOp.insert(op).second);
}

void TreeView::clearPerFrameData() { renderedMLIREntities.clear(); }

void TreeView::render() {
  render(*root, 0, false);
  renderUsers();
  renderDetachedWindows();
  pendingExpansion = false;
}

void TreeView::renderDetachedWindows() {
  if (detachedWindows.empty())
    return;

  // Windows opened while rendering wait for the next frame, avoiding
  // use-after-realloc.
  const int count = (int)detachedWindows.size();
  int dst = 0;

  for (int i = 0; i < count; ++i) {
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(detachedWindows[i].title.c_str(), &open)) {
      auto savedExpanded = std::move(mlirEntitiesPreviouslyExpanded);
      mlirEntitiesPreviouslyExpanded =
          std::move(detachedWindows[i].expandedState);
      bool savedPending = pendingExpansion;
      pendingExpansion = false;
      Operation *savedHighlight = opToHighlight;
      opToHighlight = nullptr;

      bool rc = false;
      // Copied: a push_back realloc would invalidate it.
      auto renderFn = detachedWindows[i].renderFn;
      renderFn(rc);

      // Indexing is realloc-safe.
      detachedWindows[i].expandedState =
          std::move(mlirEntitiesPreviouslyExpanded);
      mlirEntitiesPreviouslyExpanded = std::move(savedExpanded);
      pendingExpansion = savedPending;
      opToHighlight = savedHighlight;
    }
    ImGui::End();

    if (open) {
      if (dst != i)
        detachedWindows[dst] = std::move(detachedWindows[i]);
      ++dst;
    }
  }

  // Keeps windows appended during render.
  detachedWindows.erase(detachedWindows.begin() + dst,
                        detachedWindows.begin() + count);
}

void TreeView::drawElbowArrow(ImDrawList *dl, ImVec2 src, ImVec2 dst,
                              ImU32 color) {
  constexpr float kThickness = 1.5f;
  constexpr float kArrow = 6.0f;
  ImVec2 p = {std::min(src.x, dst.x) - 10.0f, src.y};
  ImVec2 q = {p.x, dst.y};
  ImVec2 arrowBase = {dst.x - kArrow, dst.y};
  dl->AddLine(src, p, color, kThickness);
  dl->AddLine(p, q, color, kThickness);
  dl->AddLine(q, arrowBase, color, kThickness);
  dl->AddTriangleFilled({dst.x, dst.y}, {dst.x - kArrow, dst.y - kArrow * 0.5f},
                        {dst.x - kArrow, dst.y + kArrow * 0.5f}, color);
}

void TreeView::renderUsers() {
  if (!opToFocus)
    return;

  const std::function<ImVec2(Operation *)> walkFromOp = [&](Operation *op) {
    auto it = renderedMLIREntities.find(op);
    if (it != renderedMLIREntities.end())
      return it->second;

    it = renderedMLIREntities.find(op->getBlock());
    if (it != renderedMLIREntities.end())
      return it->second;

    it = renderedMLIREntities.find(op->getParentRegion());
    if (it != renderedMLIREntities.end())
      return it->second;

    return walkFromOp(op->getParentOp());
  };

  const std::function<ImVec2(Block *)> walkFromBlock = [&](Block *block) {
    auto it = renderedMLIREntities.find(block);
    if (it != renderedMLIREntities.end())
      return it->second;

    it = renderedMLIREntities.find(block->getParent());
    if (it != renderedMLIREntities.end())
      return it->second;

    return walkFromOp(block->getParentOp());
  };

  // Users: focused op → each consuming op.
  constexpr ImU32 kUserColor = IM_COL32(255, 127, 0, 200);
  // Operands: each providing op → focused op.
  constexpr ImU32 kOperandColor = IM_COL32(100, 180, 255, 200);
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  ImVec2 focusPos = walkFromOp(opToFocus);

  for (auto &&user : opToFocus->getUsers())
    drawElbowArrow(drawList, focusPos, walkFromOp(user), kUserColor);

  for (auto &&operand : opToFocus->getOperands()) {
    ImVec2 providerPos;
    if (auto arg = dyn_cast<BlockArgument>(operand))
      providerPos = walkFromBlock(arg.getOwner());
    else
      providerPos = walkFromOp(operand.getDefiningOp());
    drawElbowArrow(drawList, providerPos, focusPos, kOperandColor);
  }
}

void TreeView::applyExpandState(void *entity) {
  if (pendingExpansion &&
      mlirEntitiesToExpandForHighlightedOp.contains(entity)) {
    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
  } else if (entity == mlirEntityWhereWeCollapsed) {
    mlirEntityWhereWeCollapsed = nullptr;
    ImGui::SetNextItemOpen(false, ImGuiCond_Always);
  }
}

void TreeView::render(Block &block, uint64_t id, bool recursiveCollapse) {
  applyExpandState(&block);

  auto blockLabel = llvm::formatv("{}bb{}", kIconBlock, id).str();
  auto blockOpenFn = [this, &block, id]() {
    int wid = nextDetachedId++;
    detachedWindows.push_back(DetachedWindow{
        llvm::formatv("{}bb{}###detached_{}", kIconBlock, id, wid).str(),
        [this, &block, id](bool &rc) { render(block, id, rc); }});
  };
  auto treeNode = CollapsibleTreeNode(recursiveCollapse, blockLabel.c_str(), 0,
                                      &block, std::move(blockOpenFn));
  registerEntityAsRendered(&block);

  ImGui::SameLine();
  ImGui::TextDisabled("numOps=%d numArgs=%d", (int)block.getOperations().size(),
                      (int)block.getNumArguments());

  bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&block];
  if (recursiveCollapse) {
    if (wasExpandedInThePast) {
      wasExpandedInThePast = false;
      for (auto &&[id_, op] : llvm::enumerate(block.getOperations()))
        render(op, id_, true);
    }
  } else if (treeNode) {
    wasExpandedInThePast = true;
    for (auto &&[id_, op] : llvm::enumerate(block.getOperations()))
      render(op, id_, false);
  }
}

void TreeView::render(Region &region, bool recursiveCollapse) {
  applyExpandState(&region);

  auto regionLabel =
      llvm::formatv("{}r{}", kIconRegion, region.getRegionNumber()).str();
  auto regionOpenFn = [this, &region]() {
    int wid = nextDetachedId++;
    detachedWindows.push_back(
        DetachedWindow{llvm::formatv("{}r{}###detached_{}", kIconRegion,
                                     region.getRegionNumber(), wid)
                           .str(),
                       [this, &region](bool &rc) { render(region, rc); }});
  };
  auto treeNode = CollapsibleTreeNode(recursiveCollapse, regionLabel.c_str(), 0,
                                      &region, std::move(regionOpenFn));
  registerEntityAsRendered(&region);

  ImGui::SameLine();
  ImGui::TextDisabled("numBlocks=%d", (int)region.getBlocks().size());

  bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&region];
  if (recursiveCollapse) {
    if (wasExpandedInThePast) {
      wasExpandedInThePast = false;
      for (auto &&[i, block] : llvm::enumerate(region.getBlocks()))
        render(block, i, recursiveCollapse);
    }
  } else if (treeNode) {
    wasExpandedInThePast = true;
    for (auto &&[i, block] : llvm::enumerate(region.getBlocks()))
      render(block, i, recursiveCollapse);
  }
}

void TreeView::render(Type type, bool recursiveCollapse) {
  if (recursiveCollapse)
    ImGui::SetNextItemOpen(false, ImGuiCond_Always);

  auto typeLabel =
      llvm::formatv("{}{}", kIconType, type.getAbstractType().getName()).str();
  auto typeOpenFn = [this, type]() {
    int wid = nextDetachedId++;
    detachedWindows.push_back(
        DetachedWindow{llvm::formatv("{}{}###detached_{}", kIconType,
                                     type.getAbstractType().getName(), wid)
                           .str(),
                       [this, type](bool &rc) { render(type, rc); }});
  };
  auto treeNode = CollapsibleTreeNode(recursiveCollapse, typeLabel.c_str(), 0,
                                      nullptr, std::move(typeOpenFn));
  setToolTipOnMLIREntity(type);
  if (treeNode || recursiveCollapse) {
    int id = 0;
    type.walkImmediateSubElements(
        [&](Attribute attr_) {
          ImGui::PushID(id++);
          render(attr_, recursiveCollapse);
          ImGui::PopID();
        },
        [&](Type type) {
          ImGui::PushID(id++);
          render(type, recursiveCollapse);
          ImGui::PopID();
        });
  }
}

void TreeView::render(Attribute attr, bool recursiveCollapse) {
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    ImGui::Text("%s\"%s\"", kIconAttr, strAttr.data());
  } else if (auto unitAttr = dyn_cast<UnitAttr>(attr)) {
    ImGui::Text("%sunit", kIconAttr);
  } else if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
    render(typeAttr.getValue(), recursiveCollapse);
  } else {
    if (recursiveCollapse)
      ImGui::SetNextItemOpen(false, ImGuiCond_Always);

    auto attrLabel =
        llvm::formatv("{}{}", kIconAttr, attr.getAbstractAttribute().getName())
            .str();
    auto attrOpenFn = [this, attr]() {
      int wid = nextDetachedId++;
      detachedWindows.push_back(DetachedWindow{
          llvm::formatv("{}{}###detached_{}", kIconAttr,
                        attr.getAbstractAttribute().getName(), wid)
              .str(),
          [this, attr](bool &rc) { render(attr, rc); }});
    };
    auto treeNode = CollapsibleTreeNode(recursiveCollapse, attrLabel.c_str(), 0,
                                        nullptr, std::move(attrOpenFn));
    setToolTipOnMLIREntity(attr);
    if (treeNode || recursiveCollapse) {
      int id = 0;
      attr.walkImmediateSubElements(
          [&](Attribute attr_) {
            ImGui::PushID(id++);
            render(attr_, recursiveCollapse);
            ImGui::PopID();
          },
          [&](Type type) {
            ImGui::PushID(id++);
            render(type, recursiveCollapse);
            ImGui::PopID();
          });
    }
  }
}

TreeView::CollapsibleTreeNodeGuard TreeView::ColouredCollapsibleNode(
    ImU32 colour, bool &recursiveCollapse, const char *name,
    ImGuiTreeNodeFlags flags, void *mlirEntity, std::function<void()> openFn) {
  if (recursiveCollapse)
    ImGui::SetNextItemOpen(false, ImGuiCond_Always);

  ImGui::PushStyleColor(ImGuiCol_Text, colour);
  bool wasExpanded = ImGui::TreeNodeEx(name, flags);
  // Pop before BeginPopupContextItem so the popup inherits the normal text
  // color rather than the coloured tree-node color.
  ImGui::PopStyleColor();

  bool recursiveCollapseBeforeThis = recursiveCollapse;
  if (ImGui::BeginPopupContextItem()) {
    bool canCollapse = mlirEntity != nullptr;
    ImGui::BeginDisabled(!canCollapse);
    if (ImGui::MenuItem("collapse children")) {
      mlirEntityWhereWeCollapsed = mlirEntity;
      recursiveCollapse = true;
    }
    ImGui::EndDisabled();
    if (!canCollapse &&
        ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("Not available: attributes have no unique identity "
                        "to track collapse state");
    if (openFn) {
      ImGui::Separator();
      if (ImGui::MenuItem("Open in new window"))
        openFn();
    }
    ImGui::EndPopup();
  }

  if (recursiveCollapseBeforeThis)
    ImGui::TreePush(name);

  return CollapsibleTreeNodeGuard(recursiveCollapseBeforeThis, wasExpanded);
}

void TreeView::render(DictionaryAttr dict, bool recursiveCollapse) {
  if (dict.empty())
    return;

  if (recursiveCollapse)
    ImGui::SetNextItemOpen(false, ImGuiCond_Always);

  auto attrDictLabel = llvm::formatv("{}attr-dict", kIconAttrDict).str();
  auto dictOpenFn = [this, dict]() {
    int wid = nextDetachedId++;
    detachedWindows.push_back(DetachedWindow{
        llvm::formatv("{}attr-dict###detached_{}", kIconAttrDict, wid).str(),
        [this, dict](bool &rc) { render(dict, rc); }});
  };
  auto treeNode = ColouredCollapsibleNode(
      IM_COL32(0, 255, 0, 255), recursiveCollapse, attrDictLabel.c_str(), 0,
      nullptr, std::move(dictOpenFn));
  if (treeNode || recursiveCollapse) {
    for (auto &&namedAttr : dict) {
      if (recursiveCollapse)
        ImGui::SetNextItemOpen(false, ImGuiCond_Always);

      auto namedAttrLabel =
          llvm::formatv("{}{}", kIconAttr, namedAttr.getName()).str();
      auto val = namedAttr.getValue();
      auto namedOpenFn = [this, val, name = namedAttr.getName()]() {
        int wid = nextDetachedId++;
        detachedWindows.push_back(DetachedWindow{
            llvm::formatv("{}{}###detached_{}", kIconAttr, name, wid).str(),
            [this, val](bool &rc) { render(val, rc); }});
      };
      auto treeNode =
          CollapsibleTreeNode(recursiveCollapse, namedAttrLabel.c_str(), 0,
                              nullptr, std::move(namedOpenFn));
      setToolTipOnMLIREntity(namedAttr.getValue());
      if (treeNode || recursiveCollapse)
        render(namedAttr.getValue(), recursiveCollapse);
    }
  }
}

void TreeView::render(OperandRange operands, bool recursiveCollapse) {
  renderValueRange(operands, recursiveCollapse, kIconOperands, "operands",
                   kIconOperand);
}

void TreeView::render(ResultRange results, bool recursiveCollapse) {
  renderValueRange(results, recursiveCollapse, kIconResults, "results",
                   kIconResult);
}

void TreeView::registerEntityAsRendered(void *entity) {
  ImVec2 nodeMin = ImGui::GetItemRectMin();
  ImVec2 nodeSize = ImGui::GetItemRectSize();
  ImVec2 pos = nodeMin;
  pos.y += nodeSize.y / 2;
  renderedMLIREntities[entity] = pos;
}

void TreeView::render(Operation &op, uint64_t id, bool recursiveCollapse) {
  // Ops can share names; the ID keeps hashes unique.
  ImGui::PushID(id);

  applyExpandState(&op);

  bool isOpToHighlight = &op == opToHighlight;
  ImGuiTreeNodeFlags flags = 0;
  if (isOpToHighlight) {
    if (adjustYPosToHighlightedOp && !ImGui::IsItemVisible()) {
      adjustYPosToHighlightedOp = false;
      ImGui::SetScrollHereY();
    }
    flags |= ImGuiTreeNodeFlags_Selected;
    // Slightly translucent so the node's text stays legible over the highlight.
    ImGui::PushStyleColor(ImGuiCol_Header, IM_COL32(255, 127, 0, 200));
  }

  {
    auto opLabel =
        llvm::formatv("{}{}", kIconOp, op.getName().getStringRef()).str();
    auto opOpenFn = [this, &op]() {
      int wid = nextDetachedId++;
      detachedWindows.push_back(
          DetachedWindow{llvm::formatv("{}{}###detached_{}", kIconOp,
                                       op.getName().getStringRef(), wid)
                             .str(),
                         [this, &op](bool &rc) { render(op, 0, rc); }});
    };
    auto treeNode = CollapsibleTreeNode(recursiveCollapse, opLabel.c_str(),
                                        flags, &op, std::move(opOpenFn));
    bool nodeHovered = ImGui::IsItemHovered();

    registerEntityAsRendered(&op);

    if (isOpToHighlight)
      ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::TextDisabled("numOperands=%d numResults=%d numSuccessors=%d",
                        op.getNumOperands(), op.getNumResults(),
                        op.getNumSuccessors());

    if (nodeHovered) {
      if (callbacks.scrollToOpInTextViewer)
        callbacks.scrollToOpInTextViewer(&op);
      opToFocus = &op;
    }

    bool &wasExpandedInThePast = mlirEntitiesPreviouslyExpanded[&op];

    if (recursiveCollapse) {
      // Skip recursion without open children: on huge, deeply nested graphs
      // this turns a 30 second collapse into an instant one.
      if (wasExpandedInThePast) {
        wasExpandedInThePast = false;
        render(op.getAttrDictionary(), true);
        render(op.getOperands(), true);
        render(op.getResults(), true);
        for (auto &&region : op.getRegions())
          render(region, true);
      }
    } else if (treeNode) {
      render(op.getAttrDictionary(), false);
      render(op.getOperands(), false);
      render(op.getResults(), false);

      wasExpandedInThePast = true;
      for (auto &&region : op.getRegions())
        render(region, false);
    }
  }
  ImGui::PopID();
}

TreeView::CollapsibleTreeNodeGuard
TreeView::CollapsibleTreeNode(bool &recursiveCollapse, const char *name,
                              ImGuiTreeNodeFlags flags, void *mlirEntity,
                              std::function<void()> openFn) {
  if (recursiveCollapse)
    ImGui::SetNextItemOpen(false, ImGuiCond_Always);

  bool wasExpanded = ImGui::TreeNodeEx(name, flags);
  bool recursiveCollapseBeforeThis = recursiveCollapse;
  if (ImGui::BeginPopupContextItem()) {
    bool canCollapse = mlirEntity != nullptr;
    ImGui::BeginDisabled(!canCollapse);
    if (ImGui::MenuItem("collapse children")) {
      mlirEntityWhereWeCollapsed = mlirEntity;
      recursiveCollapse = true;
    }
    ImGui::EndDisabled();
    if (!canCollapse &&
        ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("Not available: attributes have no unique identity "
                        "to track collapse state");
    if (openFn) {
      ImGui::Separator();
      if (ImGui::MenuItem("Open in new window"))
        openFn();
    }
    ImGui::EndPopup();
  }

  if (recursiveCollapseBeforeThis)
    ImGui::TreePush(name);

  return CollapsibleTreeNodeGuard(recursiveCollapseBeforeThis, wasExpanded);
}

} // namespace GlidIR
} // namespace M
