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

#include "max/max_license/API/c/detail/helpers.h"
#include "Support/Error.h"
#include "Support/ML/DType.h"
#include "max/max_license/API/c/detail/types.h" // IWYU pragma: keep (type definitions)

using namespace M;

/// Set status to error with given error.
void M::Detail::setError(M_Status *status, Error &&error) {
  status->status = std::move(error);
}

/// Set status to OK and clear message
void M::Detail::setOK(M_Status *status) { status->status = success(); }

M::DType M::Detail::mapDType(M_Dtype dtype) {
  switch (dtype) {
  case M_INT8:
    return DType::si8;
  case M_INT16:
    return DType::si16;
  case M_INT32:
    return DType::si32;
  case M_INT64:
    return DType::si64;
  case M_UINT8:
    return DType::ui8;
  case M_UINT16:
    return DType::ui16;
  case M_UINT32:
    return DType::ui32;
  case M_UINT64:
    return DType::ui64;
  case M_BOOL:
    return DType::kBool;

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, ...)                      \
  case M_TYPE:                                                                 \
    return DType::SHORT_NAME;

#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  case M_UNKNOWN:
    [[fallthrough]];
  default:
    return DType::invalid;
  }

  return DType::invalid;
}

M_Dtype M::Detail::mapDType(M::DType dtype) {
  switch (dtype.getValue()) {
  case DType::si8:
    return M_INT8;
  case DType::si16:
    return M_INT16;
  case DType::si32:
    return M_INT32;
  case DType::si64:
    return M_INT64;
  case DType::ui8:
    return M_UINT8;
  case DType::ui16:
    return M_UINT16;
  case DType::ui32:
    return M_UINT32;
  case DType::ui64:
    return M_UINT64;
  case DType::kBool:
    return M_BOOL;

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, ...)                      \
  case DType::SHORT_NAME:                                                      \
    return M_TYPE;

#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  case DType::invalid:
    [[fallthrough]];
  default:
    return M_UNKNOWN;
  }
  return M_UNKNOWN;
}
