//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TRIANGLE_PIPELINE_H
#define TRIANGLE_PIPELINE_H

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vector>
#include <webgpu/webgpu.h>

struct Vertex {
  glm::vec2 position;
  glm::vec3 color;
  uint32_t pickValue;
};

struct TrianglePipeline {
  WGPUDevice device{};
  WGPUQueue queue{};

  WGPUTexture offscreenTexture{};
  WGPUTextureView offscreenView{};

  WGPUTexture pickTexture{};
  WGPUTextureView pickView{};

  WGPURenderPipeline pipeline{};

  WGPUBuffer readBuffer{};
  // bool readMapPending = false;
  WGPUBuffer readBufferPending{};

  WGPUBuffer vertexBuffer{};

  std::vector<Vertex> vertices{};

  int width{-1};
  int height{-1};

  int pickX{-1};
  int pickY{-1};
  uint32_t pickValue{};

  TrianglePipeline(WGPUDevice device, int width, int height);
  TrianglePipeline(const TrianglePipeline &) = delete;
  TrianglePipeline &operator=(const TrianglePipeline &) = delete;
  TrianglePipeline(TrianglePipeline &&other) = delete;
  TrianglePipeline &operator=(TrianglePipeline &&other) = delete;

  ~TrianglePipeline() = default;
  void RenderTriangleToOffscreen();
  void setSize(int newWidth, int newHeight);
  void addTriangle(std::vector<Vertex> vertices);
  void addRect(float x0, float y0, float x1, float y1, glm::vec3 color,
               uint32_t pickValue = 0);

  WGPUTexture CreateOffscreenTexture(int width, int height);
  WGPUTextureView CreateTextureView(WGPUTexture texture);

  WGPUTexture CreatePickTexture(int width, int height);
  WGPUTextureView CreatePickTextureView(WGPUTexture texture);

  WGPURenderPipeline CreateTrianglePipeline();

  WGPUBuffer CreateVertexBuffer();
  void ReadPickTexture();

  void readPickValueFromReadBuffer();
};

#endif // TRIANGLE_PIPELINE_H
