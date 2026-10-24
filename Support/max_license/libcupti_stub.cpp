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
//
// On-demand CUPTI stub for GPU profiling.
//
// Tracy's GPU profiling implementation (TracyCUDA.hpp) calls CUPTI API
// functions directly.  The libkineto profiler link-references roughly fifty
// additional CUPTI symbols across activity, callbacks, events, metrics, target,
// and the Profiler API.
//
// This shared library exports every symbol both consumers need.  On first
// call each trampoline opens libcupti.so with dlopen (RTLD_NOW | RTLD_GLOBAL)
// and resolves the real implementation via dlsym, caching the function
// pointer in a function-scope static.  If the library is not present on the
// machine (no CUDA toolkit / GPU driver installed, e.g. a CPU-only
// remote-execution worker) each trampoline returns
// CUPTI_ERROR_NOT_INITIALIZED so the consumer gracefully skips GPU profiling.
//
// Linking against this stub instead of directly against libcupti.so loads CUPTI
// lazily, only when GPU profiling is actually exercised.  The companion
// libcuda_stub and libcudart_stub stubs follow the same dlopen/dlsym approach;
// new stub additions should use the macro shape below.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>

#include <cupti.h>
#include <cupti_events.h>
#include <cupti_metrics.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

namespace {

// Open libcupti.so on first use and cache the handle.  Returns nullptr if the
// library cannot be found (no CUDA toolkit / GPU driver installed).
void *getCuptiLib() {
  static void *lib = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
  return lib;
}

} // namespace

// Two-level stringize so the dlsym lookup uses the *expanded* token, matching
// the symbol that `decltype(&::NAME)` and the defined function resolve to.
// CUPTI does not version-alias its API today, but this keeps the lookup correct
// if a future header ever does, and stays symmetric with the cudart stub. For
// non-aliased names the expansion is a no-op.
#define KINETO_STUB_STR(x) #x
#define KINETO_STUB_XSTR(x) KINETO_STUB_STR(x)

// Each trampoline: dlsym the real symbol on first call (cached via static), and
// return CUPTI_ERROR_NOT_INITIALIZED if libcupti.so is missing.  Invoke it as a
// complete statement, then forward to the cached pointer:
//
//   CUPTI_STUB_TRAMPOLINE(cuptiFoo);
//   return cuptiFoo_fn(args);
#define CUPTI_STUB_TRAMPOLINE(NAME)                                            \
  using NAME##_FnType = decltype(&::NAME);                                     \
  static NAME##_FnType NAME##_fn = []()->NAME##_FnType {                       \
    void *lib = getCuptiLib();                                                 \
    return lib ? reinterpret_cast<NAME##_FnType>(                              \
                     dlsym(lib, KINETO_STUB_XSTR(NAME)))                       \
               : nullptr;                                                      \
  }                                                                            \
  ();                                                                          \
  if (!NAME##_fn)                                                              \
    return CUPTI_ERROR_NOT_INITIALIZED;

extern "C" {

// ---------------------------------------------------------------------------
// Tracy-required functions (TracyCUDA.hpp).
// ---------------------------------------------------------------------------

// cuptiGetResultString: used by Tracy's CUPTI_API_CALL error-checking macro to
// convert a CUptiResult code into a human-readable string.  Special-cased so
// the *str out-param is populated even when libcupti.so is unavailable.
CUptiResult cuptiGetResultString(CUptiResult result, const char **str) {
  using FnType = decltype(&::cuptiGetResultString);
  static FnType fn = []() -> FnType {
    void *lib = getCuptiLib();
    return lib ? reinterpret_cast<FnType>(
                     dlsym(lib, KINETO_STUB_XSTR(cuptiGetResultString)))
               : nullptr;
  }();
  if (!fn) {
    if (str)
      *str = "(CUPTI unavailable)";
    return CUPTI_ERROR_NOT_INITIALIZED;
  }
  return fn(result, str);
}

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                           CUpti_CallbackFunc callback, void *userdata) {
  CUPTI_STUB_TRAMPOLINE(cuptiSubscribe);
  return cuptiSubscribe_fn(subscriber, callback, userdata);
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityRegisterCallbacks);
  return cuptiActivityRegisterCallbacks_fn(funcBufferRequested,
                                           funcBufferCompleted);
}

CUptiResult cuptiEnableDomain(uint32_t enable,
                              CUpti_SubscriberHandle subscriber,
                              CUpti_CallbackDomain domain) {
  CUPTI_STUB_TRAMPOLINE(cuptiEnableDomain);
  return cuptiEnableDomain_fn(enable, subscriber, domain);
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityEnable);
  return cuptiActivityEnable_fn(kind);
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityDisable);
  return cuptiActivityDisable_fn(kind);
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  CUPTI_STUB_TRAMPOLINE(cuptiUnsubscribe);
  return cuptiUnsubscribe_fn(subscriber);
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityFlushAll);
  return cuptiActivityFlushAll_fn(flag);
}

CUptiResult cuptiGetCallbackName(CUpti_CallbackDomain domain, uint32_t cbid,
                                 const char **name) {
  CUPTI_STUB_TRAMPOLINE(cuptiGetCallbackName);
  return cuptiGetCallbackName_fn(domain, cbid, name);
}

CUptiResult cuptiGetTimestamp(uint64_t *timestamp) {
  CUPTI_STUB_TRAMPOLINE(cuptiGetTimestamp);
  return cuptiGetTimestamp_fn(timestamp);
}

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity **record) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityGetNextRecord);
  return cuptiActivityGetNextRecord_fn(buffer, validBufferSizeBytes, record);
}

CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context,
                                              uint32_t streamId,
                                              size_t *dropped) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityGetNumDroppedRecords);
  return cuptiActivityGetNumDroppedRecords_fn(context, streamId, dropped);
}

CUptiResult cuptiGetGraphExecId(CUgraphExec graphExec, uint32_t *pId) {
  CUPTI_STUB_TRAMPOLINE(cuptiGetGraphExecId);
  return cuptiGetGraphExecId_fn(graphExec, pId);
}

// --- libkineto-required additions follow ---

CUptiResult cuptiActivityEnableCudaEventDeviceTimestamps(uint8_t enable) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityEnableCudaEventDeviceTimestamps);
  return cuptiActivityEnableCudaEventDeviceTimestamps_fn(enable);
}

CUptiResult cuptiActivityEnableDriverApi(CUpti_CallbackId cbid,
                                         uint8_t enable) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityEnableDriverApi);
  return cuptiActivityEnableDriverApi_fn(cbid, enable);
}

CUptiResult cuptiActivityEnableRuntimeApi(CUpti_CallbackId cbid,
                                          uint8_t enable) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityEnableRuntimeApi);
  return cuptiActivityEnableRuntimeApi_fn(cbid, enable);
}

CUptiResult
cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                      uint64_t *lastId) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityPopExternalCorrelationId);
  return cuptiActivityPopExternalCorrelationId_fn(kind, lastId);
}

CUptiResult
cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                       uint64_t id) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityPushExternalCorrelationId);
  return cuptiActivityPushExternalCorrelationId_fn(kind, id);
}

CUptiResult cuptiActivityRegisterTimestampCallback(
    CUpti_TimestampCallbackFunc funcTimestamp) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivityRegisterTimestampCallback);
  return cuptiActivityRegisterTimestampCallback_fn(funcTimestamp);
}

CUptiResult cuptiActivitySetAttribute(CUpti_ActivityAttribute attr,
                                      size_t *valueSize, void *value) {
  CUPTI_STUB_TRAMPOLINE(cuptiActivitySetAttribute);
  return cuptiActivitySetAttribute_fn(attr, valueSize, value);
}

CUptiResult cuptiDeviceGetChipName(CUpti_Device_GetChipName_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiDeviceGetChipName);
  return cuptiDeviceGetChipName_fn(pParams);
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
  CUPTI_STUB_TRAMPOLINE(cuptiEnableCallback);
  return cuptiEnableCallback_fn(enable, subscriber, domain, cbid);
}

CUptiResult cuptiEventGetAttribute(CUpti_EventID event,
                                   CUpti_EventAttribute attrib,
                                   size_t *valueSize, void *value) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGetAttribute);
  return cuptiEventGetAttribute_fn(event, attrib, valueSize, value);
}

CUptiResult cuptiEventGetIdFromName(CUdevice device, const char *eventName,
                                    CUpti_EventID *event) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGetIdFromName);
  return cuptiEventGetIdFromName_fn(device, eventName, event);
}

CUptiResult cuptiEventGroupGetAttribute(CUpti_EventGroup eventGroup,
                                        CUpti_EventGroupAttribute attrib,
                                        size_t *valueSize, void *value) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupGetAttribute);
  return cuptiEventGroupGetAttribute_fn(eventGroup, attrib, valueSize, value);
}

CUptiResult cuptiEventGroupReadEvent(CUpti_EventGroup eventGroup,
                                     CUpti_ReadEventFlags flags,
                                     CUpti_EventID event,
                                     size_t *eventValueBufferSizeBytes,
                                     uint64_t *eventValueBuffer) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupReadEvent);
  return cuptiEventGroupReadEvent_fn(
      eventGroup, flags, event, eventValueBufferSizeBytes, eventValueBuffer);
}

CUptiResult cuptiEventGroupSetAttribute(CUpti_EventGroup eventGroup,
                                        CUpti_EventGroupAttribute attrib,
                                        size_t valueSize, void *value) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupSetAttribute);
  return cuptiEventGroupSetAttribute_fn(eventGroup, attrib, valueSize, value);
}

CUptiResult cuptiEventGroupSetDisable(CUpti_EventGroupSet *eventGroupSet) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupSetDisable);
  return cuptiEventGroupSetDisable_fn(eventGroupSet);
}

CUptiResult cuptiEventGroupSetEnable(CUpti_EventGroupSet *eventGroupSet) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupSetEnable);
  return cuptiEventGroupSetEnable_fn(eventGroupSet);
}

CUptiResult cuptiEventGroupSetsCreate(CUcontext context,
                                      size_t eventIdArraySizeBytes,
                                      CUpti_EventID *eventIdArray,
                                      CUpti_EventGroupSets **eventGroupPasses) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupSetsCreate);
  return cuptiEventGroupSetsCreate_fn(context, eventIdArraySizeBytes,
                                      eventIdArray, eventGroupPasses);
}

CUptiResult cuptiEventGroupSetsDestroy(CUpti_EventGroupSets *eventGroupSets) {
  CUPTI_STUB_TRAMPOLINE(cuptiEventGroupSetsDestroy);
  return cuptiEventGroupSetsDestroy_fn(eventGroupSets);
}

CUptiResult cuptiFinalize(void) {
  CUPTI_STUB_TRAMPOLINE(cuptiFinalize);
  return cuptiFinalize_fn();
}

CUptiResult cuptiGetDeviceId(CUcontext context, uint32_t *deviceId) {
  CUPTI_STUB_TRAMPOLINE(cuptiGetDeviceId);
  return cuptiGetDeviceId_fn(context, deviceId);
}

CUptiResult cuptiGetVersion(uint32_t *version) {
  CUPTI_STUB_TRAMPOLINE(cuptiGetVersion);
  return cuptiGetVersion_fn(version);
}

CUptiResult cuptiMetricEnumEvents(CUpti_MetricID metric,
                                  size_t *eventIdArraySizeBytes,
                                  CUpti_EventID *eventIdArray) {
  CUPTI_STUB_TRAMPOLINE(cuptiMetricEnumEvents);
  return cuptiMetricEnumEvents_fn(metric, eventIdArraySizeBytes, eventIdArray);
}

CUptiResult cuptiMetricGetAttribute(CUpti_MetricID metric,
                                    CUpti_MetricAttribute attrib,
                                    size_t *valueSize, void *value) {
  CUPTI_STUB_TRAMPOLINE(cuptiMetricGetAttribute);
  return cuptiMetricGetAttribute_fn(metric, attrib, valueSize, value);
}

CUptiResult cuptiMetricGetIdFromName(CUdevice device, const char *metricName,
                                     CUpti_MetricID *metric) {
  CUPTI_STUB_TRAMPOLINE(cuptiMetricGetIdFromName);
  return cuptiMetricGetIdFromName_fn(device, metricName, metric);
}

CUptiResult cuptiMetricGetNumEvents(CUpti_MetricID metric,
                                    uint32_t *numEvents) {
  CUPTI_STUB_TRAMPOLINE(cuptiMetricGetNumEvents);
  return cuptiMetricGetNumEvents_fn(metric, numEvents);
}

CUptiResult
cuptiMetricGetValue(CUdevice device, CUpti_MetricID metric,
                    size_t eventIdArraySizeBytes, CUpti_EventID *eventIdArray,
                    size_t eventValueArraySizeBytes, uint64_t *eventValueArray,
                    uint64_t timeDuration, CUpti_MetricValue *metricValue) {
  CUPTI_STUB_TRAMPOLINE(cuptiMetricGetValue);
  return cuptiMetricGetValue_fn(device, metric, eventIdArraySizeBytes,
                                eventIdArray, eventValueArraySizeBytes,
                                eventValueArray, timeDuration, metricValue);
}

CUptiResult cuptiProfilerBeginPass(CUpti_Profiler_BeginPass_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerBeginPass);
  return cuptiProfilerBeginPass_fn(pParams);
}

CUptiResult
cuptiProfilerBeginSession(CUpti_Profiler_BeginSession_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerBeginSession);
  return cuptiProfilerBeginSession_fn(pParams);
}

CUptiResult cuptiProfilerCounterDataImageCalculateScratchBufferSize(
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
        *pParams) {
  CUPTI_STUB_TRAMPOLINE(
      cuptiProfilerCounterDataImageCalculateScratchBufferSize);
  return cuptiProfilerCounterDataImageCalculateScratchBufferSize_fn(pParams);
}

CUptiResult cuptiProfilerCounterDataImageCalculateSize(
    CUpti_Profiler_CounterDataImage_CalculateSize_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerCounterDataImageCalculateSize);
  return cuptiProfilerCounterDataImageCalculateSize_fn(pParams);
}

CUptiResult cuptiProfilerCounterDataImageInitialize(
    CUpti_Profiler_CounterDataImage_Initialize_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerCounterDataImageInitialize);
  return cuptiProfilerCounterDataImageInitialize_fn(pParams);
}

CUptiResult cuptiProfilerCounterDataImageInitializeScratchBuffer(
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerCounterDataImageInitializeScratchBuffer);
  return cuptiProfilerCounterDataImageInitializeScratchBuffer_fn(pParams);
}

CUptiResult
cuptiProfilerDeInitialize(CUpti_Profiler_DeInitialize_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerDeInitialize);
  return cuptiProfilerDeInitialize_fn(pParams);
}

CUptiResult
cuptiProfilerDisableProfiling(CUpti_Profiler_DisableProfiling_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerDisableProfiling);
  return cuptiProfilerDisableProfiling_fn(pParams);
}

CUptiResult
cuptiProfilerEnableProfiling(CUpti_Profiler_EnableProfiling_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerEnableProfiling);
  return cuptiProfilerEnableProfiling_fn(pParams);
}

CUptiResult cuptiProfilerEndPass(CUpti_Profiler_EndPass_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerEndPass);
  return cuptiProfilerEndPass_fn(pParams);
}

CUptiResult cuptiProfilerEndSession(CUpti_Profiler_EndSession_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerEndSession);
  return cuptiProfilerEndSession_fn(pParams);
}

CUptiResult
cuptiProfilerFlushCounterData(CUpti_Profiler_FlushCounterData_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerFlushCounterData);
  return cuptiProfilerFlushCounterData_fn(pParams);
}

CUptiResult cuptiProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerGetCounterAvailability);
  return cuptiProfilerGetCounterAvailability_fn(pParams);
}

CUptiResult cuptiProfilerInitialize(CUpti_Profiler_Initialize_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerInitialize);
  return cuptiProfilerInitialize_fn(pParams);
}

CUptiResult cuptiProfilerPopRange(CUpti_Profiler_PopRange_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerPopRange);
  return cuptiProfilerPopRange_fn(pParams);
}

CUptiResult cuptiProfilerPushRange(CUpti_Profiler_PushRange_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerPushRange);
  return cuptiProfilerPushRange_fn(pParams);
}

CUptiResult cuptiProfilerSetConfig(CUpti_Profiler_SetConfig_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerSetConfig);
  return cuptiProfilerSetConfig_fn(pParams);
}

CUptiResult
cuptiProfilerUnsetConfig(CUpti_Profiler_UnsetConfig_Params *pParams) {
  CUPTI_STUB_TRAMPOLINE(cuptiProfilerUnsetConfig);
  return cuptiProfilerUnsetConfig_fn(pParams);
}

CUptiResult cuptiSetEventCollectionMode(CUcontext context,
                                        CUpti_EventCollectionMode mode) {
  CUPTI_STUB_TRAMPOLINE(cuptiSetEventCollectionMode);
  return cuptiSetEventCollectionMode_fn(context, mode);
}

} // extern "C"

#undef CUPTI_STUB_TRAMPOLINE
#undef KINETO_STUB_XSTR
#undef KINETO_STUB_STR
