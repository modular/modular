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
//
// Compiles the driver C interface as C, and pins every declaration in it.
//
// The point of DriverCAPI.h is that a consumer which cannot include the C++
// headers can still drive the device, and every other check on it compiles it
// as C++ (where `bool`, the opaque handle typedefs and the `extern "C"` block
// all behave differently). This translation unit is built by the CAPITest
// target, so a change that only works in C++ -- a missing <stdbool.h>, a struct
// instead of an opaque typedef, a name that collides with a C keyword -- fails
// the build rather than the consumer.
//
// Taking the address of every entry point is deliberate: assigning it to a
// function pointer of the declared type is a compile error if the return type
// or any parameter drifts, so the block below is a signature check for every
// declaration in the header, not a sample.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/CAPI/DriverCAPI.h"

int M_DriverCAPIHeaderIsConsumableFromC(void) {
  // Version.
  uint32_t (*capi_version)(void) = M_DriverCApiVersion;

  // Status.
  size_t (*last_error_message)(char *, size_t) = M_DriverLastErrorMessage;

  // Devices.
  int64_t (*accelerator_count)(void) = M_DriverAcceleratorCount;
  M_DriverStatus (*accelerator_create)(int64_t, M_DriverDevice **) =
      M_DriverAcceleratorDeviceCreate;
  M_DriverStatus (*cpu_create)(int64_t, M_DriverDevice **) =
      M_DriverCPUDeviceCreate;
  M_DriverStatus (*device_destroy)(M_DriverDevice *) = M_DriverDeviceDestroy;
  M_DriverStatus (*device_id)(const M_DriverDevice *, uint64_t *) =
      M_DriverDeviceId;
  size_t (*device_api)(const M_DriverDevice *, char *, size_t) =
      M_DriverDeviceApi;
  size_t (*device_architecture_name)(const M_DriverDevice *, char *, size_t) =
      M_DriverDeviceArchitectureName;
  size_t (*device_model_name)(const M_DriverDevice *, char *, size_t) =
      M_DriverDeviceModelName;
  size_t (*device_pci_bus_id)(const M_DriverDevice *, char *, size_t) =
      M_DriverDevicePciBusId;
  M_DriverStatus (*device_uuid)(const M_DriverDevice *, uint8_t *) =
      M_DriverDeviceUuid;
  M_DriverStatus (*device_can_access)(const M_DriverDevice *,
                                      const M_DriverDevice *, bool *) =
      M_DriverDeviceCanAccess;
  M_DriverStatus (*device_enable_peer_access)(const M_DriverDevice *,
                                              const M_DriverDevice *) =
      M_DriverDeviceEnablePeerAccess;
  M_DriverStatus (*device_synchronize)(const M_DriverDevice *) =
      M_DriverDeviceSynchronize;
  M_DriverStatus (*device_memory_info)(const M_DriverDevice *, uint64_t *,
                                       uint64_t *) = M_DriverDeviceMemoryInfo;
  M_DriverStatus (*device_allocation_range)(const M_DriverDevice *,
                                            const void *, void **, uint64_t *) =
      M_DriverDeviceAllocationRange;
  M_DriverStatus (*enable_all_peer_access)(void) = M_DriverEnableAllPeerAccess;
  M_DriverStatus (*all_peer_access_enabled)(bool *) =
      M_DriverAllPeerAccessEnabled;
  M_DriverStatus (*device_context_handle)(const M_DriverDevice *, void **) =
      M_DriverDeviceContextHandle;
  M_DriverStatus (*device_bind_to_current_thread)(const M_DriverDevice *) =
      M_DriverDeviceBindToCurrentThread;
  void (*unbind_from_current_thread)(void) = M_DriverUnbindFromCurrentThread;

  // Streams.
  M_DriverStatus (*stream_create)(const M_DriverDevice *, M_DriverStream **) =
      M_DriverStreamCreate;
  M_DriverStatus (*stream_from_context)(void *, M_DriverStream **) =
      M_DriverStreamFromContextHandle;
  M_DriverStatus (*stream_from_native)(const M_DriverDevice *, void *,
                                       M_DriverStream **) =
      M_DriverStreamFromNativeHandle;
  M_DriverStatus (*stream_destroy)(M_DriverStream *) = M_DriverStreamDestroy;
  M_DriverStatus (*stream_native_handle)(const M_DriverStream *, void **) =
      M_DriverStreamNativeHandle;
  M_DriverStatus (*stream_context_handle)(const M_DriverStream *, void **) =
      M_DriverStreamDeviceContextHandle;
  M_DriverStatus (*stream_device)(const M_DriverStream *, M_DriverDevice **) =
      M_DriverStreamDevice;
  M_DriverStatus (*stream_synchronize)(const M_DriverStream *) =
      M_DriverStreamSynchronize;
  M_DriverStatus (*stream_wait_for)(
      const M_DriverStream *, const M_DriverStream *) = M_DriverStreamWaitFor;
  M_DriverStatus (*stream_wait_for_device)(const M_DriverStream *,
                                           const M_DriverDevice *) =
      M_DriverStreamWaitForDevice;
  M_DriverStatus (*stream_wait_for_event)(const M_DriverStream *,
                                          const M_DriverEvent *) =
      M_DriverStreamWaitForEvent;
  M_DriverStatus (*stream_record_new_event)(
      const M_DriverStream *, M_DriverEvent **) = M_DriverStreamRecordNewEvent;
  M_DriverStatus (*stream_record_event)(const M_DriverStream *,
                                        const M_DriverEvent *) =
      M_DriverStreamRecordEvent;

  // Events.
  M_DriverStatus (*event_create)(const M_DriverDevice *, bool,
                                 M_DriverEvent **) = M_DriverEventCreate;
  M_DriverStatus (*event_destroy)(M_DriverEvent *) = M_DriverEventDestroy;
  M_DriverStatus (*event_synchronize)(const M_DriverEvent *) =
      M_DriverEventSynchronize;
  M_DriverStatus (*event_is_ready)(const M_DriverEvent *, bool *) =
      M_DriverEventIsReady;
  M_DriverStatus (*event_elapsed_ms)(const M_DriverEvent *,
                                     const M_DriverEvent *, double *) =
      M_DriverEventElapsedTimeMs;

  // Buffers.
  M_DriverStatus (*buffer_allocate)(const M_DriverDevice *, uint64_t, bool,
                                    M_DriverBuffer **) = M_DriverBufferAllocate;
  M_DriverStatus (*buffer_from_pointer)(const M_DriverDevice *, void *,
                                        uint64_t, bool, M_DriverBuffer **) =
      M_DriverBufferFromPointer;
  M_DriverStatus (*buffer_destroy)(M_DriverBuffer *) = M_DriverBufferDestroy;
  M_DriverStatus (*buffer_pointer)(const M_DriverBuffer *, void **) =
      M_DriverBufferPointer;
  M_DriverStatus (*buffer_byte_size)(const M_DriverBuffer *, uint64_t *) =
      M_DriverBufferByteSize;
  M_DriverStatus (*buffer_is_host)(const M_DriverBuffer *, bool *) =
      M_DriverBufferIsHost;

  // Host memory registration.
  M_DriverStatus (*memory_register)(const M_DriverDevice *, void *, uint64_t,
                                    M_DriverMapping **) =
      M_DriverMemoryRegister;
  M_DriverStatus (*mapping_destroy)(M_DriverMapping *) = M_DriverMappingDestroy;
  M_DriverStatus (*mapping_destroy_sync)(M_DriverMapping *) =
      M_DriverMappingDestroySync;
  M_DriverStatus (*mapping_pointer)(const M_DriverMapping *, void **) =
      M_DriverMappingPointer;
  M_DriverStatus (*mapping_byte_size)(const M_DriverMapping *, uint64_t *) =
      M_DriverMappingByteSize;

  // Copies.
  M_DriverStatus (*copy_host_to_device)(const M_DriverStream *, void *,
                                        const void *, uint64_t) =
      M_DriverCopyHostToDevice;
  M_DriverStatus (*copy_device_to_host)(const M_DriverStream *, void *,
                                        const void *, uint64_t) =
      M_DriverCopyDeviceToHost;
  M_DriverStatus (*copy_batch)(const M_DriverStream *, M_DriverCopyDirection,
                               void *const *, const void *const *,
                               const uint64_t *, size_t) = M_DriverCopyBatch;
  M_DriverStatus (*copy_device_to_device)(const M_DriverStream *, void *,
                                          const void *, uint64_t) =
      M_DriverCopyDeviceToDevice;
  M_DriverStatus (*copy_peer_to_peer)(
      const M_DriverStream *, const M_DriverDevice *, void *,
      const M_DriverDevice *, const void *, uint64_t) = M_DriverCopyPeerToPeer;
  M_DriverStatus (*copy_peer_to_peer_unordered)(
      const M_DriverStream *, const M_DriverDevice *, void *,
      const M_DriverDevice *, const void *, uint64_t) =
      M_DriverCopyPeerToPeerUnordered;

  // Every pointer above must have been initialized from the header, so a
  // declaration whose type changed stops compiling here.
  return capi_version != NULL && last_error_message != NULL &&
         accelerator_count != NULL && accelerator_create != NULL &&
         cpu_create != NULL && device_destroy != NULL && device_id != NULL &&
         device_api != NULL && device_architecture_name != NULL &&
         device_model_name != NULL && device_pci_bus_id != NULL &&
         device_uuid != NULL && device_can_access != NULL &&
         device_enable_peer_access != NULL && device_synchronize != NULL &&
         device_memory_info != NULL && device_allocation_range != NULL &&
         enable_all_peer_access != NULL && all_peer_access_enabled != NULL &&
         device_context_handle != NULL &&
         device_bind_to_current_thread != NULL &&
         unbind_from_current_thread != NULL && stream_create != NULL &&
         stream_from_context != NULL && stream_from_native != NULL &&
         stream_destroy != NULL && stream_native_handle != NULL &&
         stream_context_handle != NULL && stream_device != NULL &&
         stream_synchronize != NULL && stream_wait_for != NULL &&
         stream_wait_for_device != NULL && stream_wait_for_event != NULL &&
         stream_record_event != NULL && stream_record_new_event != NULL &&
         event_create != NULL && event_destroy != NULL &&
         event_synchronize != NULL && event_is_ready != NULL &&
         event_elapsed_ms != NULL && buffer_allocate != NULL &&
         buffer_from_pointer != NULL && buffer_destroy != NULL &&
         buffer_pointer != NULL && buffer_byte_size != NULL &&
         buffer_is_host != NULL && memory_register != NULL &&
         mapping_destroy != NULL && mapping_destroy_sync != NULL &&
         mapping_pointer != NULL && mapping_byte_size != NULL &&
         copy_host_to_device != NULL && copy_device_to_host != NULL &&
         copy_batch != NULL && copy_device_to_device != NULL &&
         copy_peer_to_peer != NULL && copy_peer_to_peer_unordered != NULL;
}
