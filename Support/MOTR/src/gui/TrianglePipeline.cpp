//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "TrianglePipeline.h"
#include "motr/Log.h"
#include <cmath>
#include <imgui.h>
#include <unordered_map>

namespace {
const char *triangleShaderWGSL = R"(
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) ndc: vec4<f32>,
            @location(1) color: vec4<f32>,
            @location(2) @interpolate(flat) pickData: u32,
        }

        @vertex
        fn vs_main(
            @location(0) pos: vec2<f32>,
            @location(1) color: vec3<f32>,
            @location(2) pickData: u32
        ) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(pos, 0.0, 1.0);
            output.color = vec4<f32>(color, 1.0);
            output.ndc = vec4<f32>(pos, 0.0, 1.0);
            output.pickData = pickData;

            // output.color.a = sqrt(pos.x * pos.x + pos.y * pos.y);

            return output;
        }

        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pickData: u32
        }

        @fragment
        fn fs_main(
            @builtin(position) fragCoord: vec4<f32>,
            @location(0) ndc: vec4<f32>,
            @location(1) color: vec4<f32>,
            @location(2) @interpolate(flat) pickData: u32
        ) -> FragmentOutput {
            var output: FragmentOutput;
            output.color = color;
            output.pickData = pickData;
            return output;
        }

)";
}

TrianglePipeline::TrianglePipeline(WGPUDevice device, int width, int height)
    : device(device), queue(wgpuDeviceGetQueue(device)), width(-1), height(-1) {

  pipeline = CreateTrianglePipeline();
}

int nextMultipleof64(int value) { return ((value + 63) / 64) * 64; }

void TrianglePipeline::setSize(int newWidth, int newHeight) {
  assert(newWidth > 0 && newHeight > 0);
  if (newWidth < 0 || newHeight < 0)
    return;
  if (newWidth == width && newHeight == height)
    return;
  width = newWidth;
  height = newHeight;
  {
    if (offscreenView)
      wgpuTextureViewRelease(offscreenView);
    if (offscreenTexture)
      wgpuTextureRelease(offscreenTexture);
    offscreenTexture = CreateOffscreenTexture(width, height);
    offscreenView = CreateTextureView(offscreenTexture);
  }
  {
    if (pickView)
      wgpuTextureViewRelease(pickView);
    if (pickTexture)
      wgpuTextureRelease(pickTexture);
    pickTexture = CreatePickTexture(width, height);
    pickView = CreatePickTextureView(pickTexture);
  }
  {
    if (readBuffer && readBufferPending != readBuffer)
      wgpuBufferRelease(readBuffer);
    WGPUBufferDescriptor bufferDesc = {};
    bufferDesc.size = nextMultipleof64(width) * height * sizeof(uint32_t);
    bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    readBuffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
  }
}

void TrianglePipeline::addTriangle(std::vector<Vertex> vertices) {
  this->vertices.insert(this->vertices.end(), vertices.begin(), vertices.end());
  if (vertexBuffer)
    wgpuBufferRelease(vertexBuffer);
  vertexBuffer = nullptr;
}

WGPUBuffer TrianglePipeline::CreateVertexBuffer() {
  WGPUBufferDescriptor bufDesc = {};
  bufDesc.label = "Vertex Buffer";
  bufDesc.size = vertices.size() * sizeof(Vertex);
  bufDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
  WGPUBuffer resultBuffer = wgpuDeviceCreateBuffer(device, &bufDesc);

  wgpuQueueWriteBuffer(queue, resultBuffer, 0, vertices.data(), bufDesc.size);

  return resultBuffer;
}

void TrianglePipeline::addRect(float x0, float y0, float x1, float y1,
                               glm::vec3 color, uint32_t pickValue) {
  std::vector<Vertex> rectVertices = {
      {{x0, y0}, color, pickValue}, // 0
      {{x1, y0}, color, pickValue}, // 1
      {{x1, y1}, color, pickValue}, // 2
      {{x0, y0}, color, pickValue}, // 3
      {{x1, y1}, color, pickValue}, // 4
      {{x0, y1}, color, pickValue}, // 5
  };
  addTriangle(rectVertices);
}

WGPUTexture TrianglePipeline::CreateOffscreenTexture(int width, int height) {
  WGPUTextureDescriptor desc = {};
  desc.label = "Offscreen Texture";
  desc.size.width = width;
  desc.size.height = height;
  desc.size.depthOrArrayLayers = 1;
  desc.dimension = WGPUTextureDimension_2D;
  desc.mipLevelCount = 1;
  desc.sampleCount = 1;
  desc.format = WGPUTextureFormat_BGRA8Unorm;
  desc.usage =
      WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
  return wgpuDeviceCreateTexture(device, &desc);
}

WGPUTexture TrianglePipeline::CreatePickTexture(int width, int height) {
  WGPUTextureDescriptor desc = {};
  desc.label = "Pick Texture";
  desc.size.width = width;
  desc.size.height = height;
  desc.size.depthOrArrayLayers = 1;
  desc.dimension = WGPUTextureDimension_2D;
  desc.mipLevelCount = 1;
  desc.sampleCount = 1;
  desc.format = WGPUTextureFormat_R32Uint;
  desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
  return wgpuDeviceCreateTexture(device, &desc);
}

WGPUTextureView TrianglePipeline::CreatePickTextureView(WGPUTexture texture) {
  WGPUTextureViewDescriptor viewDesc = {};
  viewDesc.label = "Pick Texture View";
  viewDesc.format = WGPUTextureFormat_R32Uint;
  viewDesc.dimension = WGPUTextureViewDimension_2D;
  viewDesc.baseMipLevel = 0;
  viewDesc.mipLevelCount = 1;
  viewDesc.baseArrayLayer = 0;
  viewDesc.arrayLayerCount = 1;
  return wgpuTextureCreateView(texture, &viewDesc);
}

WGPUTextureView TrianglePipeline::CreateTextureView(WGPUTexture texture) {
  WGPUTextureViewDescriptor viewDesc = {};
  viewDesc.label = "Texture View";
  viewDesc.format = WGPUTextureFormat_BGRA8Unorm;
  viewDesc.dimension = WGPUTextureViewDimension_2D;
  viewDesc.baseMipLevel = 0;
  viewDesc.mipLevelCount = 1;
  viewDesc.baseArrayLayer = 0;
  viewDesc.arrayLayerCount = 1;
  return wgpuTextureCreateView(texture, &viewDesc);
}

WGPURenderPipeline TrianglePipeline::CreateTrianglePipeline() {
  WGPUShaderModuleWGSLDescriptor wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgslDesc.code = triangleShaderWGSL;

  WGPUShaderModuleDescriptor smDesc = {};
  smDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct *>(&wgslDesc);

  WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &smDesc);

  WGPUVertexAttribute attributes[3] = {};
  // position
  attributes[0].shaderLocation = 0;
  attributes[0].format = WGPUVertexFormat_Float32x2;
  attributes[0].offset = offsetof(Vertex, position);
  // color
  attributes[1].shaderLocation = 1;
  attributes[1].format = WGPUVertexFormat_Float32x3;
  attributes[1].offset = offsetof(Vertex, color);
  // pick data
  attributes[2].shaderLocation = 2;
  attributes[2].format = WGPUVertexFormat_Uint32;
  attributes[2].offset = offsetof(Vertex, pickValue);

  WGPUVertexBufferLayout vertexBufferLayout = {};
  vertexBufferLayout.arrayStride = sizeof(Vertex);
  vertexBufferLayout.attributeCount = 3;
  vertexBufferLayout.attributes = attributes;
  vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;

  // color target
  WGPUColorTargetState colorTarget = {};
  colorTarget.format = WGPUTextureFormat_BGRA8Unorm;
  colorTarget.blend = nullptr;
  colorTarget.writeMask = WGPUColorWriteMask_All;

  // pick target
  WGPUColorTargetState pickTarget = {};
  pickTarget.format =
      WGPUTextureFormat_R32Uint; // Assuming 32-bit unsigned integer format
  pickTarget.blend = nullptr;
  pickTarget.writeMask = WGPUColorWriteMask_All;

  WGPUFragmentState fragmentState = {};
  fragmentState.module = shaderModule;
  fragmentState.entryPoint = "fs_main";
  fragmentState.targetCount = 2;
  WGPUColorTargetState targets[] = {colorTarget, pickTarget};
  fragmentState.targets = targets;

  WGPUVertexState vertexState = {};
  vertexState.module = shaderModule;
  vertexState.entryPoint = "vs_main";
  vertexState.bufferCount = 1;
  vertexState.buffers = &vertexBufferLayout;

  WGPUPrimitiveState primitiveState = {};
  primitiveState.topology = WGPUPrimitiveTopology_TriangleList;
  primitiveState.stripIndexFormat = WGPUIndexFormat_Undefined;
  primitiveState.frontFace = WGPUFrontFace_CCW;
  primitiveState.cullMode = WGPUCullMode_None;

  WGPUMultisampleState multisampleState = {};
  multisampleState.count = 1;
  multisampleState.mask = ~0u;
  multisampleState.alphaToCoverageEnabled = false;

  WGPURenderPipelineDescriptor pipelineDesc = {};
  pipelineDesc.label = "Triangle Pipeline";
  pipelineDesc.layout = nullptr;
  pipelineDesc.vertex = vertexState;
  pipelineDesc.primitive = primitiveState;
  pipelineDesc.fragment = &fragmentState;
  pipelineDesc.multisample = multisampleState;

  return wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
}

void TrianglePipeline::RenderTriangleToOffscreen() {
  if (vertices.empty())
    return;

  WGPUCommandEncoderDescriptor encoderDesc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(device, &encoderDesc);

  // Color attachment for offscreen rendering
  WGPURenderPassColorAttachment colorAttach = {};
  colorAttach.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
  colorAttach.view = offscreenView;
  static float time = 0.0f;
  time += 1 / 144.0f;
  // colorAttach.clearValue = {fabs(sin(time)), 0.0f, 0.0f, 1.0f};
  double grey = 0.2;
  colorAttach.clearValue = {grey, grey, grey, 1.0f};
  colorAttach.loadOp = WGPULoadOp_Clear;
  colorAttach.storeOp = WGPUStoreOp_Store;

  // Color attachment for pick rendering
  WGPURenderPassColorAttachment pickAttach = {};
  pickAttach.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
  pickAttach.view = pickView;
  pickAttach.clearValue = {0.0f, 0.0f, 0.0f,
                           0.0f}; // Clear to black or any desired value
  pickAttach.loadOp = WGPULoadOp_Clear;
  pickAttach.storeOp = WGPUStoreOp_Store;

  WGPURenderPassDescriptor rpDesc = {};
  rpDesc.colorAttachmentCount = 2;
  WGPURenderPassColorAttachment attachments[] = {colorAttach, pickAttach};
  rpDesc.colorAttachments = attachments;
  rpDesc.depthStencilAttachment = nullptr;

  WGPURenderPassEncoder pass =
      wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);

  wgpuRenderPassEncoderSetPipeline(pass, pipeline);

  if (!vertexBuffer)
    vertexBuffer = CreateVertexBuffer();

  size_t vertexBufferMemSize = vertices.size() * sizeof(Vertex);
  wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0,
                                       vertexBufferMemSize);
  wgpuRenderPassEncoderDraw(pass, vertices.size(), 1, 0, 0);

  wgpuRenderPassEncoderEnd(pass);

  WGPUCommandBufferDescriptor cmdBuffDesc = {};
  WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBuffDesc);
  wgpuQueueSubmit(queue, 1, &cmdBuffer);

  wgpuCommandBufferRelease(cmdBuffer);
  wgpuRenderPassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);

  // Optionally, wait for the queue to be idle to ensure rendering is complete
  wgpuQueueOnSubmittedWorkDone(
      queue,
      [](WGPUQueueWorkDoneStatus status, void *userData) {
        if (status == WGPUQueueWorkDoneStatus_Success) {
          // Call ReadPickTexture after rendering is complete
          static_cast<TrianglePipeline *>(userData)->ReadPickTexture();
        }
      },
      this);
}

void TrianglePipeline::readPickValueFromReadBuffer() {
  pickValue = 0;
  // if the readBufferPending is not the same as the readBuffer, we need to
  // unmap the previous buffer and release it
  if (readBufferPending != readBuffer) {
    wgpuBufferUnmap(readBufferPending);
    wgpuBufferRelease(readBufferPending);
    readBufferPending = nullptr;
    return;
  }

  if (pickX >= 0 && pickY >= 0 && pickX < width && pickY < height) {
    int width64 = nextMultipleof64(width);
    int rowPitch = width64 * sizeof(uint32_t);
    int bufferSize = rowPitch * height;
    const uint32_t *data = static_cast<const uint32_t *>(
        wgpuBufferGetConstMappedRange(readBuffer, 0, bufferSize));

    if (!data) {
      MOTR_LOG("{}", "Failed to get mapped range");
      return;
    }

    pickValue = data[pickY * width64 + pickX];
    // MOTR_LOG("Pick value: {}", pickValue);
  }

  /*
    std::unordered_map<uint32_t, uint32_t> counts;
    for (int i = 0; i < width64 * height; i++) {
      counts[data[i]]++;
    }

    double pscale = 100.0 / (width64 * height);
    for (auto &[value, count] : counts) {
      if (value == 0)
        MOTR_LOG("Pick value: {}, count: {}i ({}%)", value, count,
                 count * pscale);
    }
    */

  wgpuBufferUnmap(readBuffer);
  readBufferPending = nullptr;
}

void BufferMapCallback(WGPUBufferMapAsyncStatus status, void *userData) {
  if (status != WGPUBufferMapAsyncStatus_Success) {
    MOTR_LOG("callback failed status: {}", int(status));
    return;
  }

  static_cast<TrianglePipeline *>(userData)->readPickValueFromReadBuffer();
}

void TrianglePipeline::ReadPickTexture() {
  if (readBufferPending)
    return;
  // Create a command encoder
  WGPUCommandEncoderDescriptor encoderDesc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(device, &encoderDesc);

  // Encode the copy command
  WGPUImageCopyTexture srcTexture = {};
  srcTexture.texture = pickTexture;
  srcTexture.origin = {0, 0, 0};

  WGPUImageCopyBuffer dstBuffer = {};
  dstBuffer.buffer = readBuffer;
  dstBuffer.layout.offset = 0;

  int width64 = nextMultipleof64(this->width);

  dstBuffer.layout.bytesPerRow = width64 * sizeof(uint32_t);
  dstBuffer.layout.rowsPerImage = height;

  WGPUExtent3D copySize = {static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height), 1};
  wgpuCommandEncoderCopyTextureToBuffer(encoder, &srcTexture, &dstBuffer,
                                        &copySize);

  // Finish encoding and submit the command buffer
  WGPUCommandBufferDescriptor cmdBuffDesc = {};
  WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdBuffDesc);
  wgpuQueueSubmit(queue, 1, &cmdBuffer);

  // Release resources
  wgpuCommandBufferRelease(cmdBuffer);
  wgpuCommandEncoderRelease(encoder);

  int bufferSize = width64 * height * sizeof(uint32_t);

  // Map the buffer for reading
  readBufferPending = readBuffer;
  wgpuBufferMapAsync(readBuffer, WGPUMapMode_Read, 0, bufferSize,
                     BufferMapCallback, this);
}
