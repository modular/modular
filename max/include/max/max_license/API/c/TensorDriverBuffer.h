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
// Bridges the MAX C API to the driver C API. Internal, like DriverCAPI.h:
// the public max/c headers do not ship the driver entry points that release
// the handle.
//
//===----------------------------------------------------------------------===//

#ifndef MAX_API_C_TENSORDRIVERBUFFER_H
#define MAX_API_C_TENSORDRIVERBUFFER_H

#include "MLRT/Driver/CAPI/DriverCAPI.h"
#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Gets a new reference to the buffer backing the tensor, for the driver C
/// API's buffer view entry points, such as `M_DriverCopyBufferToHost`.
///
/// The handle shares the tensor's memory rather than copying it. Release it
/// with `M_DriverBufferDestroy`, which drops only this reference: the memory
/// stays allocated while any reference to it, the tensor's included, is held.
/// The handle is valid only with the `M_Driver*` entry points of the library
/// that returned it, never with another copy of the driver.
///
/// @param tensor The tensor.
///
/// @return The buffer handle, or `NULL` if the tensor is `NULL` or has no
/// storage or no bytes.
MODULAR_API_EXPORT M_DriverBuffer *
M_getTensorDriverBuffer(const M_AsyncTensor *tensor);

#endif // MAX_API_C_TENSORDRIVERBUFFER_H
