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

#include "Kernels/max_license/Tracing.h"
#include "nanobind/nanobind.h"
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace M::Profiler::Python {

/// Returns true if profiling is enabled via MODULAR_ENABLE_PROFILING=1 env var.
/// Truthy values include "1", "true", "on", "detailed".
bool isProfilingEnabled();

/// Sets the GPU profiling state.
void setGPUProfilingState(const char *state);

/// Adds tracing to all methods in all classes in the module
/// only if profiling is enabled via MODULAR_ENABLE_PROFILING=1 env var.
void addTracingToNanobindModuleClassMethods(nb::module_ &m, StringRef prefix);

/// Adds tracing to all methods in a class
/// only if profiling is enabled via MODULAR_ENABLE_PROFILING=1 env var
void addTracingToNanobindClassMethods(nb::object cls, StringRef prefix);

/// PyObject wrapper around a C++ interface to a Mojo Trace object.
class PyMojoTrace {
public:
  PyMojoTrace(const char *message, const char *color) {
    // Make copies of the arguments, since the Python string objects
    // may be garbage collected and destroyed before use in MojoTrace.
    messageStorage = message;
    colorStorage = color;

    ErrorOr<MojoTrace> mojoTraceOr =
        MojoTrace::create(StringRef(messageStorage), StringRef(colorStorage));
    if (mojoTraceOr.isError())
      throw std::runtime_error(mojoTraceOr.getError());

    mojoTrace = mojoTraceOr.takeValue();
  }

  PyMojoTrace &enter() {
    mojoTrace.start();
    return *this;
  }

  void exit(nb::object exc_type, nb::object exc_value, nb::object traceback) {
    mojoTrace.end();
  }

  void mark() { mojoTrace.mark(); }

private:
  std::string messageStorage;
  std::string colorStorage;
  MojoTrace mojoTrace;
};

}; // namespace M::Profiler::Python
