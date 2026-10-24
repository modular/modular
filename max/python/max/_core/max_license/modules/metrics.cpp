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

#include "Support/Metrics.h"
#include "Support/Nanobind/TypeCasters.h"
#include "_core/Graph/Bindings.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;
using namespace M::Metrics;

void M::Graph::Python::bindMetricsModule(nb::module_ m) {
  m.doc() = R"nbdoc(
      Provides in-process metrics collection for the Modular runtime.
  )nbdoc";

  nb::class_<Histogram::Snapshot>(m, "HistogramSnapshot", R"nbdoc(
      Statistics accumulated by a histogram instrument between :func:`collect()` calls.

      All fields reflect data points recorded since the previous
      :func:`collect()`.
  )nbdoc")
      .def_ro("count", &Histogram::Snapshot::count,
              "The number of data points recorded.")
      .def_ro("sum", &Histogram::Snapshot::sum,
              "The sum of all recorded values.")
      .def_ro("min", &Histogram::Snapshot::min, "The minimum recorded value.")
      .def_ro("max", &Histogram::Snapshot::max, "The maximum recorded value.");

  nb::class_<CollatedMetrics::CounterSample>(m, "CounterSample", R"nbdoc(
      A single counter reading from a :func:`collect()` snapshot.

      Counter values are cumulative and never reset.
  )nbdoc")
      .def_ro("name", &CollatedMetrics::CounterSample::name,
              "The instrument name.")
      .def_ro("value", &CollatedMetrics::CounterSample::value,
              "The cumulative count.");

  nb::class_<CollatedMetrics::GaugeSample>(m, "GaugeSample", R"nbdoc(
      A single gauge value from an instantaneous read.
  )nbdoc")
      .def_ro("name", &CollatedMetrics::GaugeSample::name,
              "The instrument name.")
      .def_ro("value", &CollatedMetrics::GaugeSample::value,
              "The current gauge value.");

  nb::class_<CollatedMetrics::HistogramSample>(m, "HistogramSample", R"nbdoc(
      A single histogram reading from a :func:`collect()` snapshot.

      The underlying accumulator is reset on each :func:`collect()` call, so
      :attr:`ss` reflects only the interval since the previous
      :func:`collect()`.
  )nbdoc")
      .def_ro("name", &CollatedMetrics::HistogramSample::name,
              "The instrument name.")
      .def_ro("ss", &CollatedMetrics::HistogramSample::ss,
              "The statistics for this collection interval.");

  nb::class_<CollatedMetrics>(m, "CollatedMetrics", R"nbdoc(
      A snapshot of all registered instruments, returned by :func:`collect()`.
  )nbdoc")
      .def_ro("counters", &CollatedMetrics::counters,
              "The counter readings, one entry per registered counter.")
      .def_ro("gauges", &CollatedMetrics::gauges,
              "The gauge readings, one entry per registered gauge.")
      .def_ro("histograms", &CollatedMetrics::histograms,
              "The histogram readings, one entry per registered histogram.");

  m.def("collect", []() { return M::Metrics::metrics().collect(); }, R"nbdoc(
      Snapshots all registered instruments.

      Histogram accumulators are cleared on each call; counter and gauge
      values are cumulative.

      Returns:
          A :class:`CollatedMetrics` containing the current reading of every
          counter and gauge, and a reset snapshot of every histogram.
  )nbdoc");
}
