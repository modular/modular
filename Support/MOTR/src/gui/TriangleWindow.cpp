//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TriangleWindow.h"
#include "imgui.h"
#include "motr/Log.h"

static int windowCount = 0;
namespace M::motr::Gui {
TriangleWindow::TriangleWindow(WGPUDevice device, const std::string &title)
    : trianglePipeline(device, 256 / 8, 256 / 8), showWindow(true),
      title(title), imguiId(fmt::format("TriangleWindow_{}", windowCount++)) {}

void TriangleWindow::createDefaultSquares() {
  trianglePipeline.addRect(0.0f, 0.0f, 1.0f, 1.0f, {1.0f, 1.0f, 0.0f});
  trianglePipeline.addRect(-1.0f, 0.0f, 0.0f, 1.0f, {0.0f, 1.0f, 0.0f});
  trianglePipeline.addRect(0.0f, -1.0f, 1.0f, 0.0f, {1.0f, 0.0f, 0.0f});
}

void TriangleWindow::renderImgui(TooltipCallback tooltipCallback) {
  if (!showWindow)
    return;

  ImGui::SetNextWindowSize(ImVec2(512, 256), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

  auto windowFlags =
      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing;

  std::string slug = fmt::format("{}##{}", title, imguiId);

  ImGui::Begin(slug.c_str(), &showWindow, windowFlags);
  {
    ImVec2 localPos = ImGui::GetWindowContentRegionMin();
    ImVec2 localPos2 = ImGui::GetCursorScreenPos();
    ImVec2 mousePos = ImGui::GetMousePos();
    ImVec2 winPos = ImGui::GetWindowPos();
    /*
    MOTR_LOG("m.y={}, l.y={}, lp2.y={}, wy={}", mousePos.y, localPos.y,
             localPos2.y, winPos.y);
    MOTR_LOG("m-l={}, m-l2={}, m-w={}, my-wp={}", mousePos.y - localPos.y,
             mousePos.y - localPos2.y, mousePos.y - winPos.y,
             localPos2.y - winPos.y);
    */

    auto pickX = mousePos.x - winPos.x - localPos.x;
    auto pickY = mousePos.y - winPos.y - localPos.y;

    // TODO: This is a hack to account for
    // imgui mouse position being offset by 5 pixels for some reason
    pickY -= 5.0;

    trianglePipeline.pickX = pickX;
    trianglePipeline.pickY = pickY;

    auto pickValue = trianglePipeline.pickValue;
    if (pickValue != 0) {
      bool visible = ImGui::IsWindowHovered(
          ImGuiHoveredFlags_AllowWhenBlockedByPopup |
          ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
      if (visible) {
        // MOTR_LOG("{} [{}, {}] -> {}", title, pickX, pickY, pickValue);
      } else {
        pickValue = 0;
      }
    }

    ImVec2 windowSize = ImGui::GetWindowSize();
    float titleBarHeight = ImGui::GetFrameHeight();
    float drawableHeight = windowSize.y - titleBarHeight;

    trianglePipeline.setSize(static_cast<int>(windowSize.x),
                             static_cast<int>(drawableHeight));
    trianglePipeline.RenderTriangleToOffscreen();

    ImGui::SetCursorPosY(titleBarHeight);
    ImGui::SetCursorPosX(0);

    ImGui::Image((ImTextureID)trianglePipeline.offscreenView,
                 ImVec2(windowSize.x, drawableHeight));

    if (tooltipCallback)
      tooltipCallback();
  }

  ImGui::End();

  ImGui::PopStyleVar(2);
}
} // namespace M::motr::Gui
