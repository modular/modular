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

#include "Event.h"
#include "Context.h"
#include "Device.h"

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_event_t>
M_driver_event::create(M_driver_context &ctx, M_driver_event_flags_t flags) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  // CUDA events are always host-queryable, so CPU_VISIBLE
  // and currently has no effect.
  (void)flags;

  CUevent event;
  if (CUresult r = cuda.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&ctx.device.driver, r, "cuEventCreate"));

  return new M_driver_event(event);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  if (CUresult r = cuda.cuEventDestroy(event->event); r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx.device.driver, r, "cuEventDestroy");

  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  if (CUresult r = cuda.cuEventSynchronize(event->event); r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx.device.driver, r, "cuEventSynchronize");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  CUresult result = cuda.cuEventQuery(event->event);
  if (result == CUDA_SUCCESS) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (result == CUDA_ERROR_NOT_READY) {
    *is_ready = false;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return cudaErrorStatus(&ctx.device.driver, result, "cuEventQuery");
}
