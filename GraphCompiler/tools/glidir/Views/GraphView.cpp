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

#include "GraphView.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <limits>

using namespace mlir;

namespace M {
namespace GlidIR {

GraphView::GraphView(ModuleOp, TextViewCallbacks &cb) : callbacks(cb) {}

void GraphView::buildGraph(ArrayRef<Operation *> ops) {
  nodes.clear();
  edges.clear();
  opIndex.clear();
  lastBookmarkSet.clear();

  mlir::OpPrintingFlags pf;
  pf.useLocalScope(false);

  for (auto &&[i, op] : llvm::enumerate(ops)) {
    lastBookmarkSet.insert(op);
    opIndex[op] = (int)i;

    Node n;
    n.op = op;
    n.name = op->getName().getStringRef().str();

    std::string s;
    llvm::raw_string_ostream os(s);
    op->print(os, pf);
    size_t start = s.find_first_not_of(" \t");
    if (start != std::string::npos)
      s = s.substr(start);
    if (s.size() > 72)
      s = s.substr(0, 69) + "\xe2\x80\xa6"; // U+2026 …
    n.text = std::move(s);

    nodes.push_back(std::move(n));
  }

  for (auto &&[dstIdx, op] : llvm::enumerate(ops)) {
    for (Value operand : op->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        auto it = opIndex.find(defOp);
        if (it != opIndex.end()) {
          std::string s;
          llvm::raw_string_ostream os(s);
          operand.printAsOperand(os, pf);
          edges.push_back({it->second, (int)dstIdx, 0, 1, 0, 1, std::move(s)});
        }
      }
    }
  }

  // Assign per-node slots so attachment points spread across node edges.
  SmallVector<int> outCount(nodes.size(), 0), inCount(nodes.size(), 0);
  for (auto &e : edges) {
    outCount[e.src]++;
    inCount[e.dst]++;
  }
  SmallVector<int> outIdx(nodes.size(), 0), inIdx(nodes.size(), 0);
  for (auto &e : edges) {
    e.srcSlot = outIdx[e.src]++;
    e.srcCount = outCount[e.src];
    e.dstSlot = inIdx[e.dst]++;
    e.dstCount = inCount[e.dst];
  }
}

void GraphView::computeLayout() {
  if (nodes.empty())
    return;
  float lh = ImGui::GetFontSize();

  // 1. Node sizes: header line + body line, capped at kMaxNodeW.
  float ellipsisW = ImGui::CalcTextSize("\xe2\x80\xa6").x;
  for (auto &n : nodes) {
    float nameW = ImGui::CalcTextSize(n.name.c_str()).x + kPadX * 2 + 16;
    float textW = ImGui::CalcTextSize(n.text.c_str()).x + kPadX * 2;
    n.w = std::min(std::max({nameW, textW, 120.0f}), kMaxNodeW);
    n.h = 2 * (lh + kPadY) + kPadY * 3;

    float maxTextW = n.w - kPadX * 2 - ellipsisW;
    const char *begin = n.text.c_str();
    const char *end = begin + n.text.size();
    const char *fit = end;
    while (fit > begin && ImGui::CalcTextSize(begin, fit).x > maxTextW) {
      --fit;
      // Step back over continuation bytes of a UTF-8 sequence.
      while (fit > begin && (*fit & 0xC0) == 0x80)
        --fit;
    }
    if (fit < end)
      n.displayText = std::string(begin, fit) + "\xe2\x80\xa6";
    else
      n.displayText = n.text;
  }

  // 2. Forward adjacency list for layering (use-def is always a DAG).
  SmallVector<SmallVector<int>> adj(nodes.size());
  for (auto &e : edges)
    adj[e.src].push_back(e.dst);

  // 3. BFS layering from roots (nodes with no incoming edges).
  SmallVector<int> inDeg(nodes.size(), 0);
  for (auto &e : edges)
    inDeg[e.dst]++;

  for (auto &n : nodes)
    n.layer = -1;

  SmallVector<int> q;
  for (int i = 0; i < (int)nodes.size(); i++)
    if (inDeg[i] == 0) {
      nodes[i].layer = 0;
      q.push_back(i);
    }
  // If everything has in-edges (cycle), seed at 0.
  if (q.empty()) {
    nodes[0].layer = 0;
    q.push_back(0);
  }
  for (int qi = 0; qi < (int)q.size(); qi++) {
    int c = q[qi];
    for (int d : adj[c]) {
      if (nodes[d].layer < nodes[c].layer + 1) {
        nodes[d].layer = nodes[c].layer + 1;
        q.push_back(d);
      }
    }
  }
  for (auto &n : nodes)
    if (n.layer < 0)
      n.layer = 0;

  // 4. Group nodes by layer and sort within each layer for stable ordering.
  int maxLayer = 0;
  for (auto &n : nodes)
    maxLayer = std::max(maxLayer, n.layer);

  std::vector<std::vector<int>> layers(maxLayer + 1);
  for (int i = 0; i < (int)nodes.size(); i++)
    layers[nodes[i].layer].push_back(i);

  // 5. Position: layers flow downward, nodes spread horizontally.
  float y = 0;
  for (auto &layer : layers) {
    float totalW = 0;
    for (int i : layer)
      totalW += nodes[i].w + kNodeGap;
    float x = -totalW / 2;
    for (int i : layer) {
      nodes[i].x = x;
      nodes[i].y = y;
      x += nodes[i].w + kNodeGap;
    }
    float maxH = 0;
    for (int i : layer)
      maxH = std::max(maxH, nodes[i].h);
    y += maxH + kLayerGap;
  }

  // 6. Single parent-alignment pass: shift each node to center over its
  //    parents (averaging parent x-centers).
  for (auto &layer : layers) {
    for (int ni : layer) {
      SmallVector<float> parentCenters;
      for (auto &e : edges) {
        if (e.dst == ni)
          parentCenters.push_back(nodes[e.src].x + nodes[e.src].w / 2);
      }
      if (parentCenters.empty())
        continue;
      float avg = 0;
      for (float c : parentCenters)
        avg += c;
      avg /= parentCenters.size();
      nodes[ni].x = avg - nodes[ni].w / 2;
    }
  }

  // 7. Resolve within-layer overlaps introduced by the alignment pass.
  //    Left-to-right sweep: push each node right of its left neighbor.
  //    Right-to-left sweep: pull nodes back left where possible.
  for (auto &layer : layers) {
    for (int k = 1; k < (int)layer.size(); ++k) {
      float minX = nodes[layer[k - 1]].x + nodes[layer[k - 1]].w + kNodeGap;
      if (nodes[layer[k]].x < minX)
        nodes[layer[k]].x = minX;
    }
    for (int k = (int)layer.size() - 2; k >= 0; --k) {
      float maxX = nodes[layer[k + 1]].x - nodes[layer[k]].w - kNodeGap;
      if (nodes[layer[k]].x > maxX)
        nodes[layer[k]].x = maxX;
    }
  }
}

void GraphView::drawContent(const DrawContext &ctx, Operation *hoveredOp,
                            Operation *selectedOp,
                            const Edge *hoveredEdge) const {
  float drawFs = ctx.lh * ctx.zoom;

  constexpr ImU32 kEdgeCol = IM_COL32(120, 150, 200, 180);
  constexpr ImU32 kEdgeHovCol = IM_COL32(255, 200, 80, 255);
  constexpr float kEW = 1.5f, kEWHov = 2.5f;

  for (const auto &e : edges) {
    const auto &src = nodes[e.src], &dst = nodes[e.dst];
    float srcCX = src.x + src.w * (e.srcSlot + 0.5f) / e.srcCount;
    float dstCX = dst.x + dst.w * (e.dstSlot + 0.5f) / e.dstCount;

    ImVec2 p0 = ctx.w(srcCX, src.y + src.h);
    ImVec2 p3 = ctx.w(dstCX, dst.y);
    float midY = (p0.y + p3.y) / 2;
    ImVec2 mid0 = {p0.x, midY};
    ImVec2 mid1 = {p3.x, midY};

    bool edgeHov = (&e == hoveredEdge);
    ImU32 col = edgeHov ? kEdgeHovCol : kEdgeCol;
    float ew = edgeHov ? kEWHov : kEW;
    ctx.dl->AddLine(p0, mid0, col, ew);
    ctx.dl->AddLine(mid0, mid1, col, ew);
    ctx.dl->AddLine(mid1, p3, col, ew);
    float a = 5.0f;
    ctx.dl->AddTriangleFilled(p3, {p3.x - a, p3.y - a * 1.6f},
                              {p3.x + a, p3.y - a * 1.6f}, col);
  }

  for (const auto &n : nodes) {
    ImVec2 p0 = ctx.w(n.x, n.y);
    ImVec2 p1 = ctx.w(n.x + n.w, n.y + n.h);

    if (p1.x < ctx.origin.x || p0.x > ctx.canvasEnd.x || p1.y < ctx.origin.y ||
        p0.y > ctx.canvasEnd.y)
      continue;

    bool isHovered = (n.op == hoveredOp && hoveredOp != nullptr);
    bool isSelected = (n.op == selectedOp && selectedOp != nullptr);
    float r = 4.0f;
    ctx.dl->AddRectFilled(p0, p1, IM_COL32(38, 47, 62, 255), r);

    float hdrH = (ctx.lh + kPadY * 2) * ctx.zoom;
    ImVec2 hdrP1 = {p1.x, p0.y + hdrH};
    ImU32 hdrCol = isHovered    ? IM_COL32(65, 105, 175, 255)
                   : isSelected ? IM_COL32(80, 70, 20, 255)
                                : IM_COL32(45, 80, 128, 255);
    ctx.dl->AddRectFilled(p0, hdrP1, hdrCol, r);
    ctx.dl->AddRectFilled({p0.x, p0.y + hdrH * 0.5f}, hdrP1, hdrCol);
    ctx.dl->AddLine({p0.x, hdrP1.y}, {p1.x, hdrP1.y},
                    IM_COL32(65, 105, 165, 255), 1.0f);
    ImU32 borderCol = isSelected  ? IM_COL32(255, 200, 60, 255)
                      : isHovered ? IM_COL32(130, 180, 255, 255)
                                  : IM_COL32(65, 110, 175, 220);
    float borderW = (isHovered || isSelected) ? 2.0f : 1.5f;
    ctx.dl->AddRect(p0, p1, borderCol, r, 0, borderW);

    if (drawFs >= 5.0f) {
      ctx.dl->PushClipRect(p0, p1, true);
      float hdrTextY = p0.y + (hdrH - drawFs) * 0.5f;
      ctx.dl->AddText(ctx.font, drawFs, {p0.x + kPadX * ctx.zoom, hdrTextY},
                      IM_COL32(200, 225, 255, 255), n.name.c_str());
      float bodyH = (p1.y - p0.y) - hdrH;
      float bodyTextY = p0.y + hdrH + (bodyH - drawFs) * 0.5f;
      ctx.dl->AddText(ctx.font, drawFs, {p0.x + kPadX * ctx.zoom, bodyTextY},
                      IM_COL32(168, 205, 162, 255), n.displayText.c_str());
      ctx.dl->PopClipRect();
    }
  }
}

GraphView::DrawContext GraphView::buildExportContext(ImDrawList *dl, int &outW,
                                                     int &outH,
                                                     float scale) const {
  const float kMargin = 20.0f * scale;

  float minX = std::numeric_limits<float>::max();
  float minY = std::numeric_limits<float>::max();
  float maxX = std::numeric_limits<float>::lowest();
  float maxY = std::numeric_limits<float>::lowest();
  for (const auto &n : nodes) {
    minX = std::min(minX, n.x);
    minY = std::min(minY, n.y);
    maxX = std::max(maxX, n.x + n.w);
    maxY = std::max(maxY, n.y + n.h);
  }
  if (nodes.empty()) {
    outW = outH = 1;
    return {dl, {0, 0}, {1, 1}, 0, 0, scale, lastLH};
  }

  float contentW = (maxX - minX) * scale;
  float contentH = (maxY - minY) * scale;
  outW = (int)(contentW + 2 * kMargin);
  outH = (int)(contentH + 2 * kMargin);
  // Pan maps world (minX, minY) to pixel (kMargin, kMargin).
  float panX = kMargin - minX * scale;
  float panY = kMargin - minY * scale;
  return {dl,     {0, 0},  {(float)outW, (float)outH}, panX, panY, scale,
          lastLH, lastFont};
}

void GraphView::render(ArrayRef<SmallVector<Operation *>> groups,
                       Operation *selectedOp) {
  // One colored tab per non-empty bookmark group.
  if (ImGui::BeginTabBar("##groups")) {
    for (int i = 0; i < (int)groups.size(); ++i) {
      if (groups[i].empty())
        continue;
      char label[64];
      snprintf(label, sizeof(label), "%s (%d)##grp%d", kBookmarkColorNames[i],
               (int)groups[i].size(), i);
      bool tabOpen = ImGui::BeginTabItem(label);
      if (tabOpen) {
        activeGroupIndex = i;
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }

  ArrayRef<Operation *> bookmarkedOps =
      (activeGroupIndex >= 0 && activeGroupIndex < (int)groups.size())
          ? ArrayRef<Operation *>(groups[activeGroupIndex])
          : ArrayRef<Operation *>{};

  DenseSet<Operation *> newSet(bookmarkedOps.begin(), bookmarkedOps.end());
  if (newSet != lastBookmarkSet)
    buildGraph(bookmarkedOps);

  ImVec2 origin = ImGui::GetCursorScreenPos();
  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (avail.x < 1 || avail.y < 1)
    return;

  ImVec2 canvasEnd = {origin.x + avail.x, origin.y + avail.y};
  lastCanvasOrigin = origin;
  lastCanvasEnd = canvasEnd;

  // Before PushClipRect, so IsItemHovered/IsItemActive work for the canvas.
  ImGui::InvisibleButton("##canvas", avail,
                         ImGuiButtonFlags_MouseButtonLeft |
                             ImGuiButtonFlags_MouseButtonRight);
  bool hov = ImGui::IsItemHovered();

  ImDrawList *dl = ImGui::GetWindowDrawList();
  dl->PushClipRect(origin, canvasEnd, true);
  dl->AddRectFilled(origin, canvasEnd, IM_COL32(22, 25, 32, 255));

  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    ImVec2 delta = ImGui::GetIO().MouseDelta;
    pan.x += delta.x;
    pan.y += delta.y;
  }
  if (hov) {
    float wheel = ImGui::GetIO().MouseWheel;
    if (wheel != 0 && (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper)) {
      float newZoom = std::clamp(zoom * (wheel > 0 ? 1.1f : 0.9f), 0.05f, 5.0f);
      // Zoom toward mouse cursor.
      ImVec2 mp = ImGui::GetMousePos();
      pan.x = mp.x - origin.x - (mp.x - origin.x - pan.x) * (newZoom / zoom);
      pan.y = mp.y - origin.y - (mp.y - origin.y - pan.y) * (newZoom / zoom);
      zoom = newZoom;
    }
  }
  if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
    pan = {300, 80};
    zoom = 1.0f;
  }

  float lh = ImGui::GetFontSize();
  lastLH = lh;
  lastFont = ImGui::GetFont();
  computeLayout();

  // For hit-testing only; drawing uses DrawContext.
  auto toScreen = [&](float wx, float wy) -> ImVec2 {
    return {origin.x + pan.x + wx * zoom, origin.y + pan.y + wy * zoom};
  };

  ImVec2 mp = ImGui::GetMousePos();
  Edge *hoveredEdge = nullptr;
  if (hov) {
    auto segDist = [](ImVec2 p, ImVec2 a, ImVec2 b) {
      ImVec2 ab = {b.x - a.x, b.y - a.y};
      float len2 = ab.x * ab.x + ab.y * ab.y;
      if (len2 == 0)
        return std::hypot(p.x - a.x, p.y - a.y);
      float t = std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2,
                           0.f, 1.f);
      ImVec2 proj = {a.x + t * ab.x, a.y + t * ab.y};
      return std::hypot(p.x - proj.x, p.y - proj.y);
    };
    constexpr float kHitRadius = 5.0f;
    for (auto &e : edges) {
      const auto &src = nodes[e.src], &dst = nodes[e.dst];
      float srcCX = src.x + src.w * (e.srcSlot + 0.5f) / e.srcCount;
      float dstCX = dst.x + dst.w * (e.dstSlot + 0.5f) / e.dstCount;
      ImVec2 p0 = toScreen(srcCX, src.y + src.h);
      ImVec2 p3 = toScreen(dstCX, dst.y);
      float midY = (p0.y + p3.y) / 2;
      ImVec2 mid0 = {p0.x, midY};
      ImVec2 mid1 = {p3.x, midY};
      if (segDist(mp, p0, mid0) < kHitRadius ||
          segDist(mp, mid0, mid1) < kHitRadius ||
          segDist(mp, mid1, p3) < kHitRadius)
        hoveredEdge = &e;
    }
  }

  // Before drawing so off-screen nodes still register hover.
  Operation *hoveredOp = nullptr;
  if (hov) {
    for (auto &n : nodes) {
      ImVec2 p0 = toScreen(n.x, n.y);
      ImVec2 p1 = toScreen(n.x + n.w, n.y + n.h);
      if (mp.x >= p0.x && mp.x < p1.x && mp.y >= p0.y && mp.y < p1.y)
        hoveredOp = n.op;
    }
  }

  DrawContext ctx{dl, origin, canvasEnd, pan.x, pan.y, zoom, lh, lastFont};
  drawContent(ctx, hoveredOp, selectedOp, hoveredEdge);

  dl->PopClipRect();

  if (hoveredEdge)
    ImGui::SetTooltip("%s", hoveredEdge->valueText.c_str());

  bool clicked = hov && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
  if (hoveredOp && callbacks.navigateToOp) {
    if (hoveredOp != lastHoveredOp || clicked)
      callbacks.navigateToOp(hoveredOp);
  }
  lastHoveredOp = hoveredOp;

  if (!pendingSave) {
    ImVec2 hint = {origin.x + 8, canvasEnd.y - ImGui::GetTextLineHeight() - 6};
    dl->AddText(hint, IM_COL32(110, 110, 110, 200),
                "\xf0\x9f\x96\xb1 drag to pan"
                "  \xc2\xb7  Ctrl+scroll to zoom"
                "  \xc2\xb7  right-click to reset");
  }
}

} // namespace GlidIR
} // namespace M
