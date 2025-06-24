//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_TRIANGLE_WINDOW_H
#define MOTR_GUI_TRIANGLE_WINDOW_H

#include "TrianglePipeline.h"
#include <functional>
#include <string>

namespace M::motr::Gui {
struct TriangleWindow {
  TrianglePipeline trianglePipeline;
  bool showWindow;
  std::string title;
  std::string imguiId;

  TriangleWindow(WGPUDevice device, const std::string &title = "");

  TriangleWindow(const TriangleWindow &) = delete;
  TriangleWindow &operator=(const TriangleWindow &) = delete;
  TriangleWindow(TriangleWindow &&other) noexcept = delete;
  TriangleWindow &operator=(TriangleWindow &&other) noexcept = delete;

  ~TriangleWindow() = default;

  void createDefaultSquares();
  using TooltipCallback = std::function<void()>;
  void renderImgui(TooltipCallback tooltipCallback);
};
} // namespace M::motr::Gui
#endif // TRIANGLE_WINDOW_H
