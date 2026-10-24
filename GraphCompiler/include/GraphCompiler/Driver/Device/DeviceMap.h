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

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/DeviceSpecs.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace M::GraphCompiler::Driver {

/// The device context backing each placement a compiled model refers to.
using DeviceMap = llvm::DenseMap<DeviceRef, M::Driver::DeviceContextRef>;

/// Returns a live, wrapped DeviceContextRef per entry in `deviceRefs`, after
/// synchronizing `deviceMap` so no driver alloc or copy is pending when the
/// first kernel runs.
///
/// `cpu` opens a fresh host context. Other placements use their `deviceMap`
/// entry, else open the accelerator at `DeviceRef::id` (erroring past the
/// physical count), which is only correct when the caller didn't supply the
/// device with the graph.
MODULAR_VISIBILITY_EXPORT ErrorOr<llvm::SmallVector<AsyncRT::AnyAsyncValueRef>>
mapDeviceRefsToDevices(const ArrayRef<DeviceRef> deviceRefs,
                       const DeviceMap &deviceMap, CPUDevice &cpuDevice);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEVICEMAP_H
