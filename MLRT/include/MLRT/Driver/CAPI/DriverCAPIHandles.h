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
// The definitions behind the driver C API's opaque handles that another C API
// in the same library hands out, so that API can wrap its own driver objects
// in them. Internal to libmax: consumers of the C interface see only the
// opaque declarations in DriverCAPI.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_CAPI_DRIVERCAPIHANDLES_H
#define MLRT_DRIVER_CAPI_DRIVERCAPIHANDLES_H

#include "MLRT/Driver/CAPI/DriverCAPI.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

/// One counted reference to a driver buffer. `M_DriverBufferDestroy` drops
/// the reference, so memory that something else also references, such as a
/// MAX tensor, stays allocated until that reference goes too.
struct M_DriverBuffer {
  M::Driver::DeviceBufferRef buffer;
};

namespace M::Driver {

/// Returns a new handle holding its own reference to `buffer`, which the
/// receiver releases with `M_DriverBufferDestroy`.
inline M_DriverBuffer *wrapDriverBuffer(const DeviceBufferRef &buffer) {
  return new M_DriverBuffer{DeviceBufferRef::copy(buffer.getPointer())};
}

} // namespace M::Driver

#endif // MLRT_DRIVER_CAPI_DRIVERCAPIHANDLES_H
