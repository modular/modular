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

#ifndef SDK_ENGINEAPI_PYTHON_DEBUGCONFIG_H
#define SDK_ENGINEAPI_PYTHON_DEBUGCONFIG_H

#include <cstdint>
#include <string>

namespace M::Core {

/// Unified debug configuration for MAX inference.
///
/// This is a thin singleton facade over the ``Support/Configuration.h``
/// ``Config`` system: setters write through ``Config::setGlobalValue()`` and
/// getters read back via ``Config::maybeGetValue()``.  Downstream components
/// that already call ``config.getValueAsBool("max-debug.<key>", ...)``
/// automatically see flags set from Python (e.g.
/// ``InferenceSession.debug.nan_check = True``) without taking a dependency on
/// the Python bindings.  Thread safety is handled by ``Config``'s own
/// override mutex; this class holds no mutable state of its own.
///
/// The ``MODULAR_DEBUG`` env var is parsed by
/// ``Config::parseModularDebugEnv()`` in ``Support``, which is invoked on
/// first ``Config::open()`` (and mirrored from this constructor for the
/// Python entry point).  Standalone Mojo binaries therefore pick up
/// ``MODULAR_DEBUG`` overrides without depending on ``max._core``.
class DebugConfig {
public:
  static DebugConfig &instance();

  // Boolean options
  bool getNanCheck() const;
  void setNanCheck(bool v);

  bool getUninitializedReadCheck() const;
  void setUninitializedReadCheck(bool v);

  bool getDeviceSyncMode() const;
  void setDeviceSyncMode(bool v);

  bool getStackTraceOnError() const;
  void setStackTraceOnError(bool v);

  bool getStackTraceOnCrash() const;
  void setStackTraceOnCrash(bool v);

  bool getSourceTracebacks() const;
  void setSourceTracebacks(bool v);

  bool getSourceTracebacksHideInternal() const;
  void setSourceTracebacksHideInternal(bool v);

  bool getPreJit() const;
  void setPreJit(bool v);

  // Value options
  std::string getUninitializedReadMode() const;
  void setUninitializedReadMode(const std::string &v);

  std::string getOpLogLevel() const;
  void setOpLogLevel(const std::string &v);

  std::string getAssertLevel() const;
  void setAssertLevel(const std::string &v);

  std::string getPrintStyle() const;
  void setPrintStyle(const std::string &v);

  std::string getIrOutputDir() const;
  void setIrOutputDir(const std::string &v);

  // libkineto / HTA / Dynolog profiler knobs (MXTOOLS-190).
  //
  // Setters write to `max-debug.profiling-*` via Config::setGlobalValue, so
  // the profiler plugin (Range.cpp) can observe them through the normal
  // Config readers without taking a Python dependency. The env-var fallbacks
  // (MODULAR_MAX_DEBUG_PROFILING_*) are synthesized automatically by
  // Config's kebab-key → ENV_VAR mapping. Knob semantics live with the
  // plugin; see docs/internal/ProfilingUserGuide.md.
  bool getProfilingEnabled() const;
  void setProfilingEnabled(bool v);

  std::string getProfilingOutputPath() const;
  void setProfilingOutputPath(const std::string &v);

  bool getProfilingDynologEnabled() const;
  void setProfilingDynologEnabled(bool v);

  int64_t getProfilingWarmupSteps() const;
  void setProfilingWarmupSteps(int64_t v);

  int64_t getProfilingActiveSteps() const;
  void setProfilingActiveSteps(int64_t v);

  int64_t getProfilingPeriodSteps() const;
  void setProfilingPeriodSteps(int64_t v);

  int64_t getProfilingPeriodicFlushSeconds() const;
  void setProfilingPeriodicFlushSeconds(int64_t v);

  // Meta mode: enables a sensible set of debug options at once.
  bool getSensibleMode() const;
  void setSensibleMode(bool v);

  // Reset all options to defaults (clears the corresponding Config overrides).
  void reset();

private:
  DebugConfig();
  DebugConfig(const DebugConfig &) = delete;
  DebugConfig &operator=(const DebugConfig &) = delete;
};

} // namespace M::Core

#endif // SDK_ENGINEAPI_PYTHON_DEBUGCONFIG_H
