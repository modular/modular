//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_RENDER_VIEW_H
#define MOTR_GUI_RENDER_VIEW_H

#include "ImGuiContext.h"
#include "RenderWindow.h"
#include <functional>
#include <map>
#include <memory>

namespace M::motr::Gui {
struct RenderViewPass;
struct RenderView;
} // namespace M::motr::Gui

struct M::motr::Gui::RenderViewPass {
  RenderViewPass(std::function<void(RenderView &)> callback, RenderView &view);
  RenderViewPass(RenderViewPass &&other) noexcept;
  RenderViewPass &operator=(RenderViewPass &&other) noexcept;
  RenderViewPass(const RenderViewPass &other) = delete;
  RenderViewPass &operator=(const RenderViewPass &other) = delete;
  ~RenderViewPass();

  void setEnabled(bool enabled);
  void setPreserve(bool preserve);
  void setZValue(int zValue);
  bool getEnabled() const;
  bool getPreserve() const;
  int getZValue() const;

  std::function<void(RenderView &)> callback;
  RenderView *view;
  bool isEnabled;
  bool preserve;
  int zValue;
};

struct M::motr::Gui::RenderView {
  RenderWindow &renderWindow;
  ImGuiContext imguiContext;

  RenderView(RenderWindow &window);

  void executeEventLoop();
  void mainLoopFunction();

  std::shared_ptr<RenderViewPass>
  addPass(std::function<void(RenderView &)> callback);
  std::shared_ptr<RenderViewPass> addPass(std::function<void()> callback);

  std::multimap<int, std::shared_ptr<RenderViewPass>> passes;
};

#endif // MOTR_GUI_RENDER_VIEW_H
