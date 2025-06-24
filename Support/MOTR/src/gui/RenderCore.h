//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef RENDER_CORE_H
#define RENDER_CORE_H

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <string>
#include <webgpu/webgpu.h>

struct RenderCore {
  WGPUDevice wgpuDevice;
  WGPUSurface wgpuSurface;
  WGPUTextureFormat wgpuPreferredFmt;
  WGPUSwapChain wgpuSwapChain;
  ImVec4 clearColor;

  RenderCore();
  RenderCore(const RenderCore &) = delete;
  RenderCore &operator=(const RenderCore &) = delete;
  RenderCore(RenderCore &&) = delete;
  RenderCore &operator=(RenderCore &&) = delete;
  ~RenderCore() = default;

  bool initWGPU(GLFWwindow *window, const std::string &css_selector);
  void createSwapChain(int width, int height);
  void handleWindowSizeChange(GLFWwindow *window);
  void postRenderLoop();
};

#endif // RENDER_CORE_H
