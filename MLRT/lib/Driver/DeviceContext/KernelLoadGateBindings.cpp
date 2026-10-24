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
// Mojo bindings for the kernel-load gate.
//
// Separate from the backend's own bindings because every entry point here
// tolerates a context with no gate, so a collective links on any backend. The
// backend bindings carry one implementation each and only one of them is
// linked at a time.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/KernelLoadGate.h"
#include "Support/SymbolExport.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;

// Exposes KernelLoadGate to a Mojo collective, through the context's driver.
//
// Returns true if the collective owns the gate now, which is always the case
// on a backend without one. Otherwise constructs the grant to dispatch from
// into `grantOut`, which the caller owns and destroys. Takes the CPUDevice
// from the current thread, since the collective's fan-out is a TaskGroup.
MODULAR_EXPORT bool AsyncRT_KernelLoadGate_tryAcquireCollective(
    Driver::DeviceContext *ctx,
    AsyncRT::AsyncValueRef<AsyncRT::Chain> *grantOut) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_KernelLoadGate_tryAcquireCollective",
                       TRACY_COLOR_GRAY);
  Driver::KernelLoadGate *gate = ctx->kernelLoadGate();
  if (gate == nullptr)
    return true;

  // Reporting success here without a claim would let the matching release
  // end some other collective's claim.
  AsyncRT::CPUDevice *cpuDevice =
      AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull();
  if (cpuDevice == nullptr)
    llvm::report_fatal_error(
        "a collective launch must run under an AsyncRT runtime");

  AsyncRT::AsyncValueRef<AsyncRT::Chain> grant;
  if (gate->tryAcquireCollective(*cpuDevice, grant))
    return true;
  new (grantOut) AsyncRT::AsyncValueRef<AsyncRT::Chain>(std::move(grant));
  return false;
}

MODULAR_EXPORT void
AsyncRT_KernelLoadGate_releaseCollective(Driver::DeviceContext *ctx) {
  if (Driver::KernelLoadGate *gate = ctx->kernelLoadGate())
    gate->releaseCollective();
}

// For vendor library calls, which can block on a busy device but never reach
// `loadFunction`.
MODULAR_EXPORT void
AsyncRT_KernelLoadGate_beginLoad(Driver::DeviceContext *ctx) {
  if (Driver::KernelLoadGate *gate = ctx->kernelLoadGate())
    gate->openLoad();
}

MODULAR_EXPORT void AsyncRT_KernelLoadGate_endLoad(Driver::DeviceContext *ctx) {
  if (Driver::KernelLoadGate *gate = ctx->kernelLoadGate())
    gate->closeLoad();
}

MODULAR_EXPORT void
AsyncRT_KernelLoadGate_beginEnqueue(Driver::DeviceContext *ctx) {
  if (Driver::KernelLoadGate *gate = ctx->kernelLoadGate())
    gate->beginEnqueue();
}

MODULAR_EXPORT void
AsyncRT_KernelLoadGate_endEnqueue(Driver::DeviceContext *ctx) {
  if (Driver::KernelLoadGate *gate = ctx->kernelLoadGate())
    gate->endEnqueue();
}
