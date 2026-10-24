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

#include "_core/Driver/ViewStrides.h"

#include <stdexcept>
#include <string>

namespace M::Driver::Python {

std::vector<ssize_t> rowMajorStrides(const TensorSpec &spec) {
  if (!spec.hasRank())
    throw std::runtime_error("unranked tensors are not supported");
  size_t rank = spec.getRank();
  std::vector<ssize_t> strides(rank);
  ssize_t offset = 1;

  for (int idx = rank - 1; idx >= 0; --idx) {
    strides[idx] = offset;
    offset *= spec[idx];
  }
  return strides;
}

std::vector<ssize_t> viewStrides(const TensorSpec &spec,
                                 const std::vector<ssize_t> &strides,
                                 bool contiguous, M::DType dtype,
                                 const std::vector<size_t> &shape,
                                 size_t byteOffset) {
  TensorSpec newSpec(shape, dtype);
  std::vector<ssize_t> newStrides = rowMajorStrides(newSpec);
  if (contiguous)
    return newStrides;

  // A view reinterprets the bytes inside a row, so the only stride it can
  // carry across is the one between rows. Dimension 0 keeps its stride and its
  // extent; anything strided inside a row, or running backwards, has no
  // expressible image in the new shape and is rejected rather than silently
  // re-addressed.
  size_t rank = spec.getRank();
  if (rank == 0)
    return newStrides;
  // A dimension of extent 1 is indexed only at 0, so its stride never applies.
  std::vector<ssize_t> packed = rowMajorStrides(spec);
  for (size_t i = 1; i < rank; ++i)
    if (spec[i] != 1 && strides[i] != packed[i])
      throw std::invalid_argument(
          "cannot view a buffer whose rows are not packed: dimension " +
          std::to_string(i) + " steps by " + std::to_string(strides[i]) +
          " where a packed row steps by " + std::to_string(packed[i]) +
          ". Only a stride on dimension 0 survives a view.");
  size_t outerDim = *spec.begin();
  // With at most one row there is no gap between rows, so the elements this
  // buffer addresses are packed and the new shape is unconstrained.
  if (outerDim <= 1)
    return newStrides;
  if (shape.empty())
    throw std::invalid_argument(
        "cannot view a strided buffer as a scalar: the stride between rows "
        "has nowhere to go");
  if (shape[0] != outerDim)
    throw std::invalid_argument(
        "a view of a strided buffer must keep dimension 0: expected " +
        std::to_string(outerDim) + ", got " + std::to_string(shape[0]) +
        ". The gap this buffer leaves between rows cannot be expressed in "
        "the new shape.");
  if (strides[0] <= 0)
    throw std::invalid_argument(
        "cannot view a buffer that steps backwards through dimension 0 "
        "(stride " +
        std::to_string(strides[0]) + ")");
  // An unaligned start rebases onto a sub-buffer sized to the view's extent,
  // which a stride reaches past.
  if (byteOffset % dtype.getSizeInBytes())
    throw std::invalid_argument(
        "cannot view a strided buffer that starts mid-element: byte offset " +
        std::to_string(byteOffset) + " is not a multiple of the view's " +
        std::to_string(dtype.getSizeInBytes()) + "-byte elements");
  // Counted in bits: a sub-byte element reports a whole byte, so bytes would
  // overstate the pitch of an fp4 or i4 buffer.
  ssize_t srcBits = spec.getEltType().getWidthInBits();
  ssize_t dstBits = dtype.getWidthInBits();
  if (srcBits <= 0 || dstBits <= 0)
    throw std::invalid_argument(
        "cannot view a strided buffer whose element width is unknown");
  ssize_t outerBits = strides[0] * srcBits;
  if (outerBits % dstBits)
    throw std::invalid_argument("the stride of a strided buffer (" +
                                std::to_string(outerBits) +
                                " bits) is not a whole number of the view's " +
                                std::to_string(dstBits) + "-bit elements");
  newStrides[0] = outerBits / dstBits;
  return newStrides;
}

} // namespace M::Driver::Python
