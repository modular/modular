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

#include "Support/RequestContext.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

void M::Graph::Python::bindRequestContextModule(nb::module_ m) {
  m.doc() = R"nbdoc(
      Which request the calling thread is currently serving.
  )nbdoc";

  m.def(
      "set_batch_id", [](int64_t batchId) { M::Request::setBatchId(batchId); },
      nb::arg("batch_id"), R"nbdoc(
      Records ``batch_id`` as the batch this thread is serving.

      C++ telemetry that opts in reads it from there — log records through
      ``MLOG_KV_REQ``, for instance. Nothing is attributed automatically: a
      call site has to ask for the id, and one that does not is unaffected.

      The context is thread-local, so work dispatched to another thread does
      not inherit it. Pair every call with :func:`clear_batch_id` through a
      ``try`` / ``finally``, so later records are not attributed to a batch
      that has already finished.
  )nbdoc");

  m.def("clear_batch_id", [] { M::Request::clearBatchId(); }, R"nbdoc(
      Clears the batch id set by :func:`set_batch_id` on this thread.

      Safe to call when no batch id is set.
  )nbdoc");
}
