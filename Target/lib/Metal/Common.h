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

#ifndef TARGET_LIB_METAL_COMMON_H
#define TARGET_LIB_METAL_COMMON_H

#include "HAL/M_driver_common.h"
#include "tl/expected.hpp"

#define M_DRIVER_STATUS_UNINITIALIZED (M_DRIVER_STATUS_CUSTOM_CODES_BASE - 0)
#define M_DRIVER_NUM_CUSTOM_STATUS_CODES 1

template <class T>
using DriverResult = tl::expected<T, M_driver_status_t>;

#endif // TARGET_LIB_METAL_COMMON_H
