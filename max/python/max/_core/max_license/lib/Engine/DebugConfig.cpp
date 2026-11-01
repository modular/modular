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

#include "_core/Engine/DebugConfig.h"
#include "Support/Configuration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace M::Core {

// DebugConfig is a thin facade over Support's Config system.  Setters write
// to a process-wide override via `Config::setGlobalValue(...)`; getters open
// a fresh Config and read back via `getValueAsBool` / `getValue`.  The Config
// override layer is itself mutex-protected, so DebugConfig holds no mutable
// state and needs no mutex of its own.
//
// Key naming: every option lives under the `max-debug.<kebab-key>` namespace,
// which `Config` auto-maps to the `MODULAR_MAX_DEBUG_<KEY>` environment
// variable for users who prefer env-var configuration.

namespace {

constexpr llvm::StringLiteral kNanCheck = "max-debug.nan-check";
constexpr llvm::StringLiteral kUninitializedReadCheck =
    "max-debug.uninitialized-read-check";
constexpr llvm::StringLiteral kDeviceSyncMode = "max-debug.device-sync-mode";
constexpr llvm::StringLiteral kStackTraceOnError =
    "max-debug.stack-trace-on-error";
constexpr llvm::StringLiteral kStackTraceOnCrash =
    "max-debug.stack-trace-on-crash";
constexpr llvm::StringLiteral kSourceTracebacks = "max-debug.source-tracebacks";
constexpr llvm::StringLiteral kSourceTracebacksHideInternal =
    "max-debug.source-tracebacks-hide-internal";
constexpr llvm::StringLiteral kPreJit = "max-debug.pre-jit";
constexpr llvm::StringLiteral kUninitializedReadMode =
    "max-debug.uninitialized-read-mode";
constexpr llvm::StringLiteral kOpLogLevel = "max-debug.op-log-level";
constexpr llvm::StringLiteral kAssertLevel = "max-debug.assert-level";
constexpr llvm::StringLiteral kPrintStyle = "max-debug.print-style";
constexpr llvm::StringLiteral kIrOutputDir = "max-debug.ir-output-dir";
constexpr llvm::StringLiteral kSensibleMode = "max-debug.sensible-mode";

// libkineto / HTA / Dynolog profiler keys (MXTOOLS-190).
constexpr llvm::StringLiteral kProfilingEnabled = "max-debug.profiling-enabled";
constexpr llvm::StringLiteral kProfilingOutputPath =
    "max-debug.profiling-output-path";
constexpr llvm::StringLiteral kProfilingDynologEnabled =
    "max-debug.profiling-dynolog-enabled";
constexpr llvm::StringLiteral kProfilingWarmupSteps =
    "max-debug.profiling-warmup-steps";
constexpr llvm::StringLiteral kProfilingActiveSteps =
    "max-debug.profiling-active-steps";
constexpr llvm::StringLiteral kProfilingPeriodSteps =
    "max-debug.profiling-period-steps";
constexpr llvm::StringLiteral kProfilingPeriodicFlushSeconds =
    "max-debug.profiling-periodic-flush-seconds";

const char *boolStr(bool v) { return v ? "true" : "false"; }

bool readBool(llvm::StringRef key, bool defaultValue = false) {
  auto configOr = M::Config::open();
  if (configOr.isError())
    return defaultValue;
  return configOr->getValueAsBool(key, defaultValue);
}

std::string readString(llvm::StringRef key) {
  auto configOr = M::Config::open();
  if (configOr.isError())
    return "";
  return configOr->getValue(key).str();
}

// Emits the parse-failure warning at most once per key. getters are read on
// every access (e.g. potentially once per model.execute()), so an unguarded
// warning on a misconfigured env var would spam stderr; this keeps it to a
// single visible line per misconfigured key.
static void warnOncePerKey(llvm::StringRef key, llvm::StringRef value,
                           int64_t defaultValue) {
  static std::mutex mu;
  static std::set<std::string> warned;
  std::lock_guard<std::mutex> lock(mu);
  if (!warned.insert(key.str()).second)
    return;
  llvm::errs() << "warning: DebugConfig: failed to parse '" << key << "' = \""
               << value << "\" as int64; using default " << defaultValue
               << "\n";
}

// Config has no typed int reader, so we go through getValue() and parse.
// On parse failure (empty / non-numeric) we return defaultValue rather than
// raise — matches the lenient semantics of readBool() above. A non-empty
// value that fails to parse emits a one-line stderr warning (once per key,
// see warnOncePerKey) so a non-numeric env var (e.g.
// MODULAR_MAX_DEBUG_PROFILING_ACTIVE_STEPS=ten) is at least visible instead
// of silently reverting to the default.
int64_t readInt64(llvm::StringRef key, int64_t defaultValue) {
  auto configOr = M::Config::open();
  if (configOr.isError())
    return defaultValue;
  llvm::StringRef value = configOr->getValue(key);
  if (value.empty())
    return defaultValue;
  int64_t parsed = 0;
  if (value.getAsInteger(/*Radix=*/10, parsed)) {
    warnOncePerKey(key, value, defaultValue);
    return defaultValue;
  }
  return parsed;
}

} // namespace

DebugConfig &DebugConfig::instance() {
  static DebugConfig inst;
  return inst;
}

DebugConfig::DebugConfig() { Config::parseModularDebugEnv(); }

// --- Boolean getters/setters ---

bool DebugConfig::getNanCheck() const { return readBool(kNanCheck); }
void DebugConfig::setNanCheck(bool v) {
  M::Config::setGlobalValue(kNanCheck, boolStr(v));
}

bool DebugConfig::getUninitializedReadCheck() const {
  return readBool(kUninitializedReadCheck);
}
void DebugConfig::setUninitializedReadCheck(bool v) {
  M::Config::setGlobalValue(kUninitializedReadCheck, boolStr(v));
}

bool DebugConfig::getDeviceSyncMode() const {
  return readBool(kDeviceSyncMode);
}
void DebugConfig::setDeviceSyncMode(bool v) {
  M::Config::setGlobalValue(kDeviceSyncMode, boolStr(v));
}

bool DebugConfig::getStackTraceOnError() const {
  return readBool(kStackTraceOnError);
}
void DebugConfig::setStackTraceOnError(bool v) {
  M::Config::setGlobalValue(kStackTraceOnError, boolStr(v));
}

bool DebugConfig::getStackTraceOnCrash() const {
  return readBool(kStackTraceOnCrash);
}
void DebugConfig::setStackTraceOnCrash(bool v) {
  M::Config::setGlobalValue(kStackTraceOnCrash, boolStr(v));
}

bool DebugConfig::getSourceTracebacks() const {
  return readBool(kSourceTracebacks);
}
void DebugConfig::setSourceTracebacks(bool v) {
  M::Config::setGlobalValue(kSourceTracebacks, boolStr(v));
}

bool DebugConfig::getSourceTracebacksHideInternal() const {
  return readBool(kSourceTracebacksHideInternal, /*defaultValue=*/true);
}
void DebugConfig::setSourceTracebacksHideInternal(bool v) {
  M::Config::setGlobalValue(kSourceTracebacksHideInternal, boolStr(v));
}

bool DebugConfig::getPreJit() const { return readBool(kPreJit); }
void DebugConfig::setPreJit(bool v) {
  M::Config::setGlobalValue(kPreJit, boolStr(v));
}

// --- Value getters/setters ---

std::string DebugConfig::getUninitializedReadMode() const {
  return readString(kUninitializedReadMode);
}
void DebugConfig::setUninitializedReadMode(const std::string &v) {
  M::Config::setGlobalValue(kUninitializedReadMode, v);
}

std::string DebugConfig::getOpLogLevel() const {
  return readString(kOpLogLevel);
}
void DebugConfig::setOpLogLevel(const std::string &v) {
  M::Config::setGlobalValue(kOpLogLevel, v);
}

std::string DebugConfig::getAssertLevel() const {
  return readString(kAssertLevel);
}
void DebugConfig::setAssertLevel(const std::string &v) {
  M::Config::setGlobalValue(kAssertLevel, v);
}

std::string DebugConfig::getPrintStyle() const {
  return readString(kPrintStyle);
}
void DebugConfig::setPrintStyle(const std::string &v) {
  M::Config::setGlobalValue(kPrintStyle, v);
}

std::string DebugConfig::getIrOutputDir() const {
  return readString(kIrOutputDir);
}
void DebugConfig::setIrOutputDir(const std::string &v) {
  M::Config::setGlobalValue(kIrOutputDir, v);
}

// --- libkineto / HTA / Dynolog profiler knobs (MXTOOLS-190) ---

bool DebugConfig::getProfilingEnabled() const {
  return readBool(kProfilingEnabled);
}
void DebugConfig::setProfilingEnabled(bool v) {
  M::Config::setGlobalValue(kProfilingEnabled, boolStr(v));
}

std::string DebugConfig::getProfilingOutputPath() const {
  return readString(kProfilingOutputPath);
}
void DebugConfig::setProfilingOutputPath(const std::string &v) {
  M::Config::setGlobalValue(kProfilingOutputPath, v);
}

// Defaults to true so fleet-wide Dynolog collection works out of the box,
// matching the ProfilingConfig pydantic default.
bool DebugConfig::getProfilingDynologEnabled() const {
  return readBool(kProfilingDynologEnabled, /*defaultValue=*/true);
}
void DebugConfig::setProfilingDynologEnabled(bool v) {
  M::Config::setGlobalValue(kProfilingDynologEnabled, boolStr(v));
}

// Both step-window knobs share one contract with the plugin (Range.cpp):
// 0 (also the unset default) means "no window" — enable() starts collection
// immediately and the capture runs until disable(). A nonzero active-steps
// default here would silently truncate every capture that leaves the knobs
// unset.
int64_t DebugConfig::getProfilingWarmupSteps() const {
  return readInt64(kProfilingWarmupSteps, /*defaultValue=*/0);
}
void DebugConfig::setProfilingWarmupSteps(int64_t v) {
  if (v < 0)
    throw std::invalid_argument("profiling_warmup_steps must be >= 0 (got " +
                                std::to_string(v) + ")");
  M::Config::setGlobalValue(kProfilingWarmupSteps, std::to_string(v));
}

int64_t DebugConfig::getProfilingActiveSteps() const {
  return readInt64(kProfilingActiveSteps, /*defaultValue=*/0);
}
void DebugConfig::setProfilingActiveSteps(int64_t v) {
  if (v < 0)
    throw std::invalid_argument("profiling_active_steps must be >= 0 (got " +
                                std::to_string(v) + "); 0 means unbounded");
  M::Config::setGlobalValue(kProfilingActiveSteps, std::to_string(v));
}

// 0 (the unset default) means one window. A nonzero period repeats the
// active window every `v` steps; the plugin falls back to one window unless
// it exceeds active-steps.
int64_t DebugConfig::getProfilingPeriodSteps() const {
  return readInt64(kProfilingPeriodSteps, /*defaultValue=*/0);
}
void DebugConfig::setProfilingPeriodSteps(int64_t v) {
  if (v < 0)
    throw std::invalid_argument("profiling_period_steps must be >= 0 (got " +
                                std::to_string(v) + "); 0 means one window");
  M::Config::setGlobalValue(kProfilingPeriodSteps, std::to_string(v));
}

int64_t DebugConfig::getProfilingPeriodicFlushSeconds() const {
  return readInt64(kProfilingPeriodicFlushSeconds, /*defaultValue=*/60);
}
void DebugConfig::setProfilingPeriodicFlushSeconds(int64_t v) {
  // Mirror ProfilingConfig's `ge=1` bound: a flush cadence below one second is
  // nonsensical and pydantic rejects it, so reject it here too.
  if (v < 1)
    throw std::invalid_argument(
        "profiling_periodic_flush_seconds must be >= 1 (got " +
        std::to_string(v) + ")");
  M::Config::setGlobalValue(kProfilingPeriodicFlushSeconds, std::to_string(v));
}

// --- Meta mode ---

bool DebugConfig::getSensibleMode() const { return readBool(kSensibleMode); }

void DebugConfig::setSensibleMode(bool v) {
  M::Config::setGlobalValue(kSensibleMode, boolStr(v));
  if (v) {
    setNanCheck(true);
    setAssertLevel("all");
    setDeviceSyncMode(true);
    setStackTraceOnError(true);
    setStackTraceOnCrash(true);
    setSourceTracebacks(true);
  }
}

// --- Reset ---

// reset() clears every Config override this facade set.  This leaves the
// underlying env vars and config-file values untouched, so a reset returns
// the process to its "pristine" state rather than forcing every option to
// `false`/`""`.
void DebugConfig::reset() {
  M::Config::unsetGlobalValue(kNanCheck);
  M::Config::unsetGlobalValue(kUninitializedReadCheck);
  M::Config::unsetGlobalValue(kDeviceSyncMode);
  M::Config::unsetGlobalValue(kStackTraceOnError);
  M::Config::unsetGlobalValue(kStackTraceOnCrash);
  M::Config::unsetGlobalValue(kSourceTracebacks);
  M::Config::unsetGlobalValue(kSourceTracebacksHideInternal);
  M::Config::unsetGlobalValue(kPreJit);
  M::Config::unsetGlobalValue(kUninitializedReadMode);
  M::Config::unsetGlobalValue(kOpLogLevel);
  M::Config::unsetGlobalValue(kAssertLevel);
  M::Config::unsetGlobalValue(kPrintStyle);
  M::Config::unsetGlobalValue(kIrOutputDir);
  M::Config::unsetGlobalValue(kSensibleMode);
  M::Config::unsetGlobalValue(kProfilingEnabled);
  M::Config::unsetGlobalValue(kProfilingOutputPath);
  M::Config::unsetGlobalValue(kProfilingDynologEnabled);
  M::Config::unsetGlobalValue(kProfilingWarmupSteps);
  M::Config::unsetGlobalValue(kProfilingActiveSteps);
  M::Config::unsetGlobalValue(kProfilingPeriodSteps);
  M::Config::unsetGlobalValue(kProfilingPeriodicFlushSeconds);
}

} // namespace M::Core
