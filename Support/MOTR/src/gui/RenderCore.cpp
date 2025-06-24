//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RenderCore.h"
#include <webgpu/webgpu_cpp.h>
#ifdef __EMSCRIPTEN__
#include <emscripten/html5_webgpu.h>
#endif

#include "imgui_impl_wgpu.h"

static void wgpu_error_callback(WGPUErrorType error_type, const char *message,
                                void *) {
  const char *error_type_lbl = "";
  switch (error_type) {
  case WGPUErrorType_Validation:
    error_type_lbl = "Validation";
    break;
  case WGPUErrorType_OutOfMemory:
    error_type_lbl = "Out of memory";
    break;
  case WGPUErrorType_Unknown:
    error_type_lbl = "Unknown";
    break;
  case WGPUErrorType_DeviceLost:
    error_type_lbl = "Device lost";
    break;
  default:
    error_type_lbl = "Unknown";
  }
  printf("%s error: %s\n", error_type_lbl, message);
}

#ifndef __EMSCRIPTEN__
static WGPUAdapter RequestAdapter(WGPUInstance instance) {
  auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status,
                                  WGPUAdapter adapter, const char *message,
                                  void *pUserData) {
    if (status == WGPURequestAdapterStatus_Success)
      *(WGPUAdapter *)(pUserData) = adapter;
    else
      printf("Could not get WebGPU adapter: %s\n", message);
  };
  WGPUAdapter adapter;
  wgpuInstanceRequestAdapter(instance, nullptr, onAdapterRequestEnded,
                             (void *)&adapter);
  return adapter;
}

static WGPUDevice RequestDevice(WGPUAdapter &adapter) {
  auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status,
                                 WGPUDevice device, const char *message,
                                 void *pUserData) {
    if (status == WGPURequestDeviceStatus_Success)
      *(WGPUDevice *)(pUserData) = device;
    else
      printf("Could not get WebGPU device: %s\n", message);
  };
  WGPUDevice device;
  wgpuAdapterRequestDevice(adapter, nullptr, onDeviceRequestEnded,
                           (void *)&device);
  return device;
}
#endif

RenderCore::RenderCore()
    : wgpuDevice(nullptr), wgpuSurface(nullptr),
      wgpuPreferredFmt(WGPUTextureFormat_BGRA8Unorm), wgpuSwapChain(nullptr),
      clearColor(ImVec4(0.25f, 0.25f, 0.25f, 1.00f)) {
  //
}

bool RenderCore::initWGPU(GLFWwindow *window, const std::string &css_selector) {
  wgpu::Instance instance = wgpuCreateInstance(nullptr);

#ifdef __EMSCRIPTEN__
  wgpuDevice = emscripten_webgpu_get_device();
  if (!wgpuDevice)
    return false;
#else
  WGPUAdapter adapter = RequestAdapter(instance.Get());
  if (!adapter)
    return false;
  wgpuDevice = RequestDevice(adapter);
#endif

#ifdef __EMSCRIPTEN__
  wgpu::SurfaceDescriptorFromCanvasHTMLSelector html_surface_desc = {};
  html_surface_desc.selector = css_selector.c_str();
  wgpu::SurfaceDescriptor surface_desc = {};
  surface_desc.nextInChain = &html_surface_desc;
  wgpu::Surface surface = instance.CreateSurface(&surface_desc);

  wgpu::Adapter adapter = {};
#else
  wgpu::Surface surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);
  if (!surface)
    return false;
  wgpuPreferredFmt = WGPUTextureFormat_BGRA8Unorm;
#endif

  wgpuSurface = surface.MoveToCHandle();

  wgpuDeviceSetUncapturedErrorCallback(wgpuDevice, wgpu_error_callback,
                                       nullptr);

  return true;
}

void RenderCore::createSwapChain(int width, int height) {
  if (wgpuSwapChain)
    wgpuSwapChainRelease(wgpuSwapChain);
  WGPUSwapChainDescriptor swap_chain_desc = {};
  swap_chain_desc.usage = WGPUTextureUsage_RenderAttachment;
  swap_chain_desc.format = wgpuPreferredFmt;
  swap_chain_desc.width = width;
  swap_chain_desc.height = height;
  swap_chain_desc.presentMode = WGPUPresentMode_Fifo;
  wgpuSwapChain =
      wgpuDeviceCreateSwapChain(wgpuDevice, wgpuSurface, &swap_chain_desc);
}

void RenderCore::handleWindowSizeChange(GLFWwindow *window) {
  int width;
  int height;
  static int last_width = 0;
  static int last_height = 0;
  glfwGetFramebufferSize(window, &width, &height);

  // todo: this is a hack to fix the issue where the swap chain is not
  //       created when the window is resized
  // Error: Viewport bounds (x: 0.000000, y: 0.000000, width: 1477.000122,
  // height: 1611.000000) are not contained in the render target dimensions
  // (1477 x 1611). While encoding [RenderPassEncoder
  // (unlabeled)].SetViewport(0.000000, 0.000000, 1477.000122, 1611.000000,
  // 0.000000, 1.000000). While finishing [CommandEncoder (unlabeled)].0
  width += 1;
  height += 1;

  if (width == last_width && height == last_height)
    return;

  ImGui_ImplWGPU_InvalidateDeviceObjects();
  createSwapChain(width, height);
  ImGui_ImplWGPU_CreateDeviceObjects();
  last_width = width;
  last_height = height;
}

void RenderCore::postRenderLoop() {
#ifndef __EMSCRIPTEN__
  wgpuDeviceTick(wgpuDevice);
#endif

  WGPURenderPassColorAttachment color_attachments = {};
  color_attachments.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
  color_attachments.loadOp = WGPULoadOp_Clear;
  color_attachments.storeOp = WGPUStoreOp_Store;
  color_attachments.clearValue = {clearColor.x * clearColor.w,
                                  clearColor.y * clearColor.w,
                                  clearColor.z * clearColor.w, clearColor.w};
  color_attachments.view = wgpuSwapChainGetCurrentTextureView(wgpuSwapChain);

  WGPURenderPassDescriptor render_pass_desc = {};
  render_pass_desc.colorAttachmentCount = 1;
  render_pass_desc.colorAttachments = &color_attachments;
  render_pass_desc.depthStencilAttachment = nullptr;

  WGPUCommandEncoderDescriptor enc_desc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(wgpuDevice, &enc_desc);

  WGPURenderPassEncoder pass =
      wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_desc);
  ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBufferDescriptor cmd_buffer_desc = {};
  WGPUCommandBuffer cmd_buffer =
      wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
  WGPUQueue queue = wgpuDeviceGetQueue(wgpuDevice);
  wgpuQueueSubmit(queue, 1, &cmd_buffer);

#ifndef __EMSCRIPTEN__
  wgpuSwapChainPresent(wgpuSwapChain);
#endif

  wgpuTextureViewRelease(color_attachments.view);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);
  wgpuCommandBufferRelease(cmd_buffer);
}
