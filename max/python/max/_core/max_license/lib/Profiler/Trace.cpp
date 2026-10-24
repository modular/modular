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

#include "_core/Profiler/Trace.h"
#include "Kernels/max_license/Tracing.h"
#include "Support/StringExtras.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Process.h"
#include <string>

namespace nb = nanobind;

using namespace M;

/// External profiler annotation override in KGEN CompilerRT, declared weak
/// so this library keeps working when CompilerRT is absent from the
/// link/load closure (the annotation integration then keeps its
/// environment default).
extern "C" __attribute__((weak)) void
KGEN_CompilerRT_ExternalProfilerAnnotationSetEnabled(size_t enabled);

static std::string getDefaultProfilingState() {
  static const char *envName = "MODULAR_ENABLE_PROFILING";
  return llvm::sys::Process::GetEnv(envName).value_or("off");
}

/// Returns the state of the GPU profiling.
static std::string &getGPUProfilingState() {
  static std::string state = getDefaultProfilingState();
  return state;
}

/// Returns true if the environment variable MODULAR_ENABLE_PROFILING is set to
/// a truthy value such as "1", "true", "on", "detailed".
bool M::Profiler::Python::isProfilingEnabled() {
  std::string &state = getGPUProfilingState();
  if (isTrueLike(state))
    return true;
  return state == "detailed";
}

void M::Profiler::Python::setGPUProfilingState(const char *state) {
  std::string &currentState = getGPUProfilingState();
  currentState = state;
  // Keep the external profiler annotation gate in step with the Python-side
  // gate, so a programmatic enable also emits vendor annotations (and a
  // disable suppresses them) without touching MODULAR_ENABLE_PROFILING.
  if (&KGEN_CompilerRT_ExternalProfilerAnnotationSetEnabled != nullptr)
    KGEN_CompilerRT_ExternalProfilerAnnotationSetEnabled(
        isProfilingEnabled() ? 1 : 0);
}

void M::Profiler::Python::addTracingToNanobindModuleClassMethods(
    nb::module_ &m, StringRef prefix) {
  if (!isProfilingEnabled())
    return;

  // nb::enum_ produces a Python enum class, whose dict holds callables that
  // are not methods (`_member_type_` is `object`, `__new__` and the operators
  // are the enum machinery's own). Wrapping those breaks it, e.g. combining
  // Flag members raises TypeError. Enums nested in a class are left alone by
  // the per-class sweep below, which never wraps a type.
  nb::handle enumType = nb::module_::import_("enum").attr("EnumType");
  nb::dict moduleDict = m.attr("__dict__");
  for (auto [key, value] : moduleDict) {
    if (value.is_type() && !PyObject_IsInstance(value.ptr(), enumType.ptr())) {
      std::string modulePrefix =
          prefix.str() + nb::borrow<nb::str>(key).c_str() + ".";
      addTracingToNanobindClassMethods(nb::cast<nb::object>(value),
                                       modulePrefix);
    }
  }
}

std::pair<std::string, int> getPyScopeSourceLoc() {
  nb::gil_scoped_acquire gil;
  nb::object frame = nb::module_::import_("inspect").attr("currentframe")();

  // Get caller's frame
  nb::object callerFrame = frame.attr("f_back");
  if (!callerFrame.is_none()) {
    nb::object filename = callerFrame.attr("f_code").attr("co_filename");
    nb::object lineno = callerFrame.attr("f_lineno");
    return {nb::cast<std::string>(filename), nb::cast<int>(lineno)};
  }

  return {"unknown", 0};
}

// Wrap all methods of a nb::class_ with tracing.
void M::Profiler::Python::addTracingToNanobindClassMethods(nb::object cls,
                                                           StringRef prefix) {
  if (!isProfilingEnabled())
    return;

  nb::mapping classDict = cls.attr("__dict__");
  for (const auto item : classDict.items()) {
    auto key = item[0];
    auto value = item[1];
    // A type in a class dict is a nested class or enum, not a method;
    // replacing it with a function breaks attribute access through it.
    if (!PyCallable_Check(value.ptr()) || PyType_Check(value.ptr()))
      continue;

    std::string functionName = nb::cast<std::string>(key);
    // HACK: disable tracing of __dlpack__ method, for some reason these
    // spans never get terminated.
    // TODO(PAQ-1463): Remove this hack once the underlying issue is resolved.
    if (functionName == "__dlpack__")
      continue;

    std::string traceName = prefix.str() + functionName;
    nb::object originalFunc = nb::cast<nb::object>(value);

    /// Create a lambda wrapper that traces the original function.
    auto traced = [originalFunc, traceName](nb::args args, nb::kwargs kwargs) {
      ErrorOr<MojoTrace> trace = M::MojoTrace::create(traceName.c_str());
      if (!trace.isError())
        trace->start();

      nb::object result = originalFunc(*args, **kwargs);
      if (!trace.isError())
        trace->end();
      return result;
    };

    /// Create a py function that wraps the trace original function
    /// Note that this approach incurs a small runtime overhead
    /// because it creates two new function levels of indirection.
    /// But it allows us to runtime trace the original function only when
    /// profiling is enabled via MODULAR_ENABLE_PROFILING = 1
    auto tracedPyFunc = nb::cpp_function(traced, nb::name(functionName.c_str()),
                                         nb::is_method());

    // Replace the method in the class
    cls.attr(functionName.c_str()) = tracedPyFunc;
  }
}
