//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RenderView.h"
#include "ImGuiContext.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_wgpu.h"
#include "imgui.h"
#include "motr/Log.h"
#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

using namespace M;
using namespace M::motr::Gui;

RenderView::RenderView(RenderWindow &window)
    : renderWindow(window), imguiContext(window) {}

std::shared_ptr<RenderViewPass>
RenderView::addPass(std::function<void(RenderView &)> callback) {
  auto pass = std::make_shared<RenderViewPass>(std::move(callback), *this);
  passes.emplace(0, pass);
  return pass;
}

std::shared_ptr<RenderViewPass>
RenderView::addPass(std::function<void()> callback) {
  return addPass(
      [callback = std::move(callback)](RenderView &view) { callback(); });
}

void RenderView::mainLoopFunction() {
  glfwPollEvents();

#ifndef __EMSCRIPTEN__
  if (glfwGetWindowAttrib(renderWindow.window, GLFW_ICONIFIED) != 0) {
    ImGui_ImplGlfw_Sleep(10);
    return;
  }
#endif

  renderWindow.renderCore.handleWindowSizeChange(renderWindow.window);

  ImGui_ImplWGPU_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // Top-level invisible fullscreen window
  ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGuiWindowFlags windowFlags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoNavFocus |
      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("##TopLevel", nullptr, windowFlags);
  ImGui::PopStyleVar(3);

  for (auto &[zValue, pass] : passes)
    if (pass->getEnabled() && pass->callback)
      pass->callback(*this);

  ImGui::End();

  ImGui::Render();
  renderWindow.renderCore.postRenderLoop();
}

extern "C" void callMainLoopFunction(void *arg) {
  static_cast<RenderView *>(arg)->mainLoopFunction();
}

void RenderView::executeEventLoop() {
#ifdef __EMSCRIPTEN__
  emscripten_set_main_loop_arg(callMainLoopFunction, this, 0, true);
#else
  while (!glfwWindowShouldClose(renderWindow.window))
    mainLoopFunction();
#endif
}

// RenderViewPass implementation
RenderViewPass::RenderViewPass(std::function<void(RenderView &)> callback,
                               RenderView &view)
    : callback(std::move(callback)), view(&view), isEnabled(true),
      preserve(false), zValue(0) {}

RenderViewPass::RenderViewPass(RenderViewPass &&other) noexcept
    : callback(std::move(other.callback)), view(other.view),
      isEnabled(other.isEnabled), preserve(other.preserve),
      zValue(other.zValue) {
  other.view = nullptr;
}

RenderViewPass &RenderViewPass::operator=(RenderViewPass &&other) noexcept {
  if (this != &other) {
    callback = std::move(other.callback);
    view = other.view;
    isEnabled = other.isEnabled;
    preserve = other.preserve;
    zValue = other.zValue;
    other.view = nullptr;
  }
  return *this;
}

RenderViewPass::~RenderViewPass() {
  if (!preserve && view) {
    for (auto it = view->passes.begin(); it != view->passes.end(); ++it) {
      if (it->second.get() == this) {
        view->passes.erase(it);
        break;
      }
    }
  }
}

void RenderViewPass::setEnabled(bool enabled) { isEnabled = enabled; }
void RenderViewPass::setPreserve(bool preserve) { this->preserve = preserve; }
void RenderViewPass::setZValue(int zValue) { this->zValue = zValue; }
bool RenderViewPass::getEnabled() const { return isEnabled; }
bool RenderViewPass::getPreserve() const { return preserve; }
int RenderViewPass::getZValue() const { return zValue; }
