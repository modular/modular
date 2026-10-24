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

#include "Queue.h"

#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Event.h"
#include "Function.h"
#include "Memory.h"
#include <algorithm>

DriverResult<M_driver_queue_t>
M_driver_queue::create(M_driver_context_t context) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(context->device->mDevice->newCommandQueue());
  if (!commandQueue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  NS::SharedPtr<MTL::Fence> orderFence =
      NS::TransferPtr(context->device->mDevice->newFence());
  if (!orderFence)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  MTL::CommandBuffer *primer = commandQueue->commandBuffer();
  MTL::BlitCommandEncoder *blit =
      primer ? primer->blitCommandEncoder() : nullptr;
  if (!blit)
    return tl::unexpected(context->device->driver->status.create(
        "Failed to create command buffer to prime the ordering fence"));
  blit->updateFence(orderFence.get());
  blit->endEncoding();
  primer->commit();

  M_driver_queue_t queue = new (std::nothrow) M_driver_queue{
      context,          commandQueue,      orderFence,   /*mDonatedWaits=*/{},
      /*mCommitSeq=*/0, /*mDrainedSeq=*/0, /*mMutex=*/{}};
  if (!queue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return queue;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

void M_driver_queue::encodeDonatedWaits(MTL::CommandBuffer *cb) {
  for (auto it = mDonatedWaits.begin(); it != mDonatedWaits.end();) {
    bool satisfied =
        it->sharedView
            ? (it->sharedView->signaledValue() >= it->value)
            : (it->retired && it->retired->load(std::memory_order_acquire));
    if (satisfied) {
      it = mDonatedWaits.erase(it);
      continue;
    }
    cb->encodeWait(it->event.get(), it->value);
    if (!it->sharedView && !it->handlerInstalled) {
      std::shared_ptr<std::atomic<bool>> flag = it->retired;
      cb->addCompletedHandler(^(MTL::CommandBuffer *) {
        flag->store(true, std::memory_order_release);
      });
      it->handlerInstalled = true;
    }
    ++it;
  }
}

M_driver_status_t M_driver_queue::drainPriorWork() {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::CommandBuffer *cmdBuffer = nullptr;
  uint64_t seq = 0;
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mCommitSeq == mDrainedSeq)
      return M_DRIVER_STATUS_SUCCESS;
    seq = mCommitSeq;
    cmdBuffer = mCommandQueue->commandBuffer();
    if (cmdBuffer == nullptr)
      return context->device->driver->status.create(
          "Failed to create command buffer for drain");
    encodeDonatedWaits(cmdBuffer);
    MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
    if (blit == nullptr)
      return context->device->driver->status.create(
          "Failed to create blit encoder for drain");
    blit->waitForFence(mOrderingFence.get());
    blit->endEncoding();
    cmdBuffer->commit();
  }
  cmdBuffer->waitUntilCompleted();
  {
    std::lock_guard<std::mutex> lock(mMutex);
    mDrainedSeq = std::max(mDrainedSeq, seq);
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
  //                      TRACY_COLOR_PURPLE);

  if (src == nullptr)
    return context->device->driver->status.create("Source pointer is null");
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  dstOffset += dst->offset;

  NS::UInteger bufferLength = dst->mBuffer->length();
  if (bufferLength == 0)
    return context->device->driver->status.create(
        "Metal buffer length is 0 - buffer may be invalid");

  // Check that copy fits within the buffer from the offset
  if (dstOffset + size > bufferLength)
    return context->device->driver->status.create(
        "Copy size exceeds Metal buffer length");

  // Pinned host memory is one of this context's MTLBuffers, so it blits without
  // waiting, like cuMemcpyHtoDAsync. Pageable memory stays synchronous below.
  if (auto pinned = context->getBufferWithOffset(src); pinned.isValid()) {
    if (pinned.offset + size > pinned.buffer->length())
      return context->device->driver->status.create(
          "Copy size exceeds pinned source buffer length");
    return enqueueBlit(dst->mBuffer, dstOffset, pinned.buffer, pinned.offset,
                       size);
  }

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // The transfer is a host memcpy: drain prior GPU work, then copy inline.
  if (M_driver_status_t status = drainPriorWork();
      status != M_DRIVER_STATUS_SUCCESS)
    return status;
  std::memcpy(static_cast<char *>(dstContents) + dstOffset, src, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
  //                      TRACY_COLOR_PURPLE);
  if (dst == nullptr)
    return context->device->driver->status.create(
        "Destination pointer cannot be null");
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  srcOffset += src->offset;

  if (srcOffset + size > src->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal source buffer length");

  if (auto pinned = context->getBufferWithOffset(dst); pinned.isValid()) {
    if (pinned.offset + size > pinned.buffer->length())
      return context->device->driver->status.create(
          "Copy size exceeds pinned destination buffer length");
    return enqueueBlit(pinned.buffer, pinned.offset, src->mBuffer, srcOffset,
                       size);
  }

  void *srcContents = src->mBuffer->contents();
  if (srcContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // Drain the work producing `src`, then read inline.
  if (M_driver_status_t status = drainPriorWork();
      status != M_DRIVER_STATUS_SUCCESS)
    return status;
  std::memcpy(dst, static_cast<char *>(srcContents) + srcOffset, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  dstOffset += dst->offset;
  srcOffset += src->offset;

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal destination buffer length");
  if (srcOffset + size > src->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal source buffer length");

  return enqueueBlit(dst->mBuffer, dstOffset, src->mBuffer, srcOffset, size);
}

M_driver_status_t M_driver_queue::enqueueBlit(MTL::Buffer *dst,
                                              uint64_t dstOffset,
                                              MTL::Buffer *src,
                                              uint64_t srcOffset,
                                              uint64_t size) {
  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodeDonatedWaits(cmdBuffer);
  MTL::BlitCommandEncoder *encoder = cmdBuffer->blitCommandEncoder();
  if (encoder == nullptr)
    return context->device->driver->status.create(
        "Failed to create blit command encoder");
  encoder->copyFromBuffer(src, srcOffset, dst, dstOffset, size);
  encoder->updateFence(mOrderingFence.get());
  encoder->endEncoding();

  cmdBuffer->commit();
  mCommitSeq += 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t dstOffset, uint64_t size,
                                            uint8_t value) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
  //                      TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  dstOffset += dst->offset;

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Set size exceeds Metal buffer length");

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dstOffset;

  // A host memset into shared memory (see copyToDevice): drain, then write.
  if (M_driver_status_t status = drainPriorWork();
      status != M_DRIVER_STATUS_SUCCESS)
    return status;
  std::memset(dstPtr, value, size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemory(dst, dstOffset, size, static_cast<uint8_t>(value));

  if (valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  if ((size % valueSize) != 0)
    return M_DRIVER_STATUS_INVALID_ARG;

  dstOffset += dst->offset;

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Fill size exceeds Metal buffer length");

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dstOffset;

  size_t count = size / valueSize;
  if (reinterpret_cast<uintptr_t>(dstPtr) % valueSize != 0)
    return context->device->driver->status.create(
        "destination buffer not properly aligned");

  // A host fill into shared memory (see copyToDevice): drain, then write.
  if (M_driver_status_t status = drainPriorWork();
      status != M_DRIVER_STATUS_SUCCESS)
    return status;
  switch (valueSize) {
  case 2: {
    uint16_t v16 = static_cast<uint16_t>(value);
    uint16_t *p16 = static_cast<uint16_t *>(dstPtr);
    for (size_t i = 0; i < count; ++i)
      p16[i] = v16;
    break;
  }
  case 4: {
    uint32_t v32 = static_cast<uint32_t>(value);
    uint32_t *p32 = static_cast<uint32_t *>(dstPtr);
    for (size_t i = 0; i < count; ++i)
      p32[i] = v32;
    break;
  }
  case 8: {
    uint64_t *p64 = static_cast<uint64_t *>(dstPtr);
    for (size_t i = 0; i < count; ++i)
      p64[i] = value;
    break;
  }
  }
  return M_DRIVER_STATUS_SUCCESS;
}

namespace {
MTL::Size toSize(M_driver_dim_t dim) { return {dim.x, dim.y, dim.z}; }
} // namespace

M_driver_status_t
M_driver_queue::execute(M_driver_function_t function,
                        M_driver_queue_execute_config_t *config, void **args,
                        uint64_t *argSizes, uint32_t numArgs) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueFunctionExecDirect",
  //                      TRACY_COLOR_PURPLE);

  if (config->mode != M_DRIVER_QUEUE_EXECUTE_MODE_GPU)
    return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Validate every argument before opening an encoder. Metal asserts when a
  // command buffer is dropped with an open encoder, so it is not safe to bail
  // out partway through encoding — check all arguments up front.
  const NS::UInteger maxBufferLength =
      context->device->mDevice->maxBufferLength();
  for (uint32_t i = 0; i < function->argumentCount; ++i) {
    if (args[i] == nullptr)
      return context->device->driver->status.create("Null argument at index " +
                                                    std::to_string(i));
    if (argSizes[i] == 0)
      return context->device->driver->status.create(
          "Unknown argument size at index " + std::to_string(i));
    // Reject per-argument buffers at or above the device's per-buffer size
    // limit, mirroring the guard on context allocations. Checked here, before
    // the encoder is opened, so the per-arg newBuffer() below cannot return
    // null mid-encode.
    if (argSizes[i] >= maxBufferLength)
      return context->device->driver->status.create(
          "Failed to allocate MTL Buffer of a size " +
          std::to_string(argSizes[i]) + " bytes which is larger than allowed " +
          std::to_string(maxBufferLength) + " bytes");
  }

  // Validate the dynamic threadgroup (shared) memory request against the device
  // limit before opening an encoder — Metal asserts if a command buffer is
  // dropped with an open encoder, so reject early.
  const uint32_t sharedMemBytes = config->config.gpu.shared_mem_bytes;
  if (sharedMemBytes > 0) {
    NS::UInteger maxSharedMemBytes =
        context->device->mDevice->maxThreadgroupMemoryLength();
    if (sharedMemBytes > maxSharedMemBytes)
      return context->device->driver->status.create(
          "Shared memory byte count " + std::to_string(sharedMemBytes) +
          " exceeds device limit of " + std::to_string(maxSharedMemBytes));
  }

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodeDonatedWaits(cmdBuffer);
  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    return context->device->driver->status.create(
        "Failed to create compute command encoder");
  encoder->setComputePipelineState(function->pipelineState.get());

  // Bind all buffer arguments. Arguments were validated above, before the
  // encoder was opened.
  for (uint32_t i = 0; i < function->argumentCount; ++i) {
    // We have more complex argument, such as pointer to a struct. Allocate
    // new MTLBuffer and set it.
    NS::SharedPtr<MTL::Buffer> buf =
        NS::TransferPtr(context->device->mDevice->newBuffer(
            args[i], argSizes[i],
            MTL::ResourceOptions(MTL::StorageModeShared) |
                MTL::ResourceOptions(MTL::CPUCacheModeWriteCombined)));
    encoder->setBuffer(buf.get(), 0, i);
  }

  // Set resources that might be used by the kernel
  // FIXME: only set resources that are used by the kernel
  for (const auto &it : context->allocations)
    encoder->useResource(it.second.get(), MTL::ResourceUsageWrite);

  // Bind dynamic threadgroup (shared) memory if the kernel requested it.
  if (sharedMemBytes > 0)
    encoder->setThreadgroupMemoryLength(sharedMemBytes, /*index=*/0);

  // Dispatch
  MTL::Size gridSize = toSize(config->config.gpu.grid);
  MTL::Size threadgroupSize = toSize(config->config.gpu.block);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);

  encoder->updateFence(mOrderingFence.get());
  encoder->endEncoding();

  cmdBuffer->commit();
  mCommitSeq += 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() { return drainPriorWork(); }

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer for record_event");

  // Donated waits + the fence wait gather everything committed before us.
  encodeDonatedWaits(cmdBuffer);
  MTL::BlitCommandEncoder *blit = cmdBuffer->blitCommandEncoder();
  if (blit == nullptr)
    return context->device->driver->status.create(
        "Failed to create blit encoder for record_event");
  blit->waitForFence(mOrderingFence.get());
  blit->endEncoding();

  // Reserve after the fallible steps so a failure cannot strand the value.
  uint64_t previous =
      event->targetValue.fetch_add(1, std::memory_order_acq_rel);
  uint64_t reserved = previous + 1;
  // Chain so a cross-queue re-record cannot signal ahead of an earlier one.
  if (previous > 0)
    cmdBuffer->encodeWait(event->getEvent(), previous);
  cmdBuffer->encodeSignalEvent(event->getEvent(), reserved);
  cmdBuffer->commit();
  mCommitSeq += 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t numEvents) {
  if (numEvents == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  std::lock_guard<std::mutex> lock(mMutex);

  // Capture each external's value now (CUDA capture-at-call semantics).
  for (uint32_t i = 0; i < numEvents; ++i) {
    uint64_t target = events[i]->targetValue.load(std::memory_order_acquire);
    // A never-recorded event is complete (CUDA semantics).
    if (target == 0)
      continue;
    DonatedWait wait;
    if (events[i]->sharedEvent) {
      wait.event = NS::SharedPtr<MTL::Event>(events[i]->sharedEvent);
      wait.sharedView = events[i]->sharedEvent.get();
    } else {
      wait.event = events[i]->deviceEvent;
      wait.retired = std::make_shared<std::atomic<bool>>(false);
    }
    wait.value = target;
    mDonatedWaits.push_back(std::move(wait));
    mCommitSeq += 1;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  auto queueOr = M_driver_queue::create(context);
  if (!queueOr)
    return queueOr.error();
  *queue = *queueOr;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue_destroy([[maybe_unused]] M_driver_context_t context,
                       M_driver_queue_t queue) {
  return M_driver_queue::destroy(queue);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->copyToDevice(dst->memory, dst->offset, src, dst->size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyFromDevice(dst, src->memory, src->offset, src->size);
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyIntraDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size);
}

M_driver_status_t M_driver_queue_copy_inter_device(
    [[maybe_unused]] M_driver_queue_t queue,
    [[maybe_unused]] M_driver_memory_view_t dst,
    [[maybe_unused]] M_driver_memory_view_t src,
    [[maybe_unused]] M_driver_context_t src_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->setMemory(dst->memory, dst->offset, dst->size, value);
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->fillMemory(dst->memory, dst->offset, dst->size, value,
                           value_size);
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config != nullptr && config->config.gpu.num_attributes > 0)
    return M_DRIVER_STATUS_UNIMPLEMENTED;
  return queue->execute(function, config, args, arg_sizes, num_args);
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->recordEvent(event);
}

M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (events == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->waitForEvents(events, num_events);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->synchronize();
}

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (llvm::StringRef(property_name) == M_DRIVER_QUEUE_PROPERTY_NATIVE_HANDLE) {
    *static_cast<uint64_t *>(value) =
        reinterpret_cast<uint64_t>(queue->mCommandQueue.get());
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (is_stream == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = false;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue_launch_host_func([[maybe_unused]] M_driver_queue_t queue,
                                [[maybe_unused]] void (*fn)(void *),
                                [[maybe_unused]] void *user_data) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_queue_wait_value64([[maybe_unused]] M_driver_queue_t queue,
                            [[maybe_unused]] uint64_t device_address,
                            [[maybe_unused]] uint64_t value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
