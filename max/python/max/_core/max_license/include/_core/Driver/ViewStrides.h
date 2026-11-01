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

#ifndef SDK_DRIVERAPI_PYTHON_VIEWSTRIDES_H
#define SDK_DRIVERAPI_PYTHON_VIEWSTRIDES_H

#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include <cstddef>
#include <sys/types.h>
#include <vector>

namespace M::Driver::Python {

/// Returns the strides of a packed row-major `spec`, counted in elements.
std::vector<ssize_t> rowMajorStrides(const TensorSpec &spec);

/// Returns the strides a view of `shape` and `dtype` takes on, given a source
/// buffer of `spec` and `strides` starting `byteOffset` bytes into its storage.
///
/// A view reinterprets the bytes inside a row, so the only stride it can carry
/// across is the one between rows. Throws `std::invalid_argument` when the
/// source's stride has no image in the new shape.
std::vector<ssize_t> viewStrides(const TensorSpec &spec,
                                 const std::vector<ssize_t> &strides,
                                 bool contiguous, M::DType dtype,
                                 const std::vector<size_t> &shape,
                                 size_t byteOffset);

} // namespace M::Driver::Python

#endif // SDK_DRIVERAPI_PYTHON_VIEWSTRIDES_H
