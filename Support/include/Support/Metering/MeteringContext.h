//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METERING_METERINGCONTEXT_H
#define SUPPORT_METERING_METERINGCONTEXT_H

#include <chrono>
#include <condition_variable>
#include <thread>

#include "llvm/ADT/StringRef.h"

#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Telemetry.h"
#include "Support/Threading/Atomics.h"

namespace M::Metering {

/// Governs periodic metering operations (e.g. billing) that export through
/// telemetry events.
///
/// Periodically uploads on a background thread or, if the added thread is
/// undesirable, flush() can be explicitly called.
class MeteringContext {
public:
  using ClockType = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<ClockType>;
  using DurationType = std::chrono::duration<ClockType::rep, ClockType::period>;

  // Called every interval with elapsed seconds + shutdown status.
  using MeterCallbackFn = std::function<ErrorOrSuccess(DurationType, bool)>;
  using MeterAttributes = llvm::StringMap<M::Telemetry::Logs::AttributeValue>;

  // Telemetry attribute names/values.
  static constexpr StringLiteral kEventDomain = "metering";
  static constexpr StringLiteral kEventName = "meter";
  static constexpr StringLiteral kEventType = "cpu_usage_v1";

  static constexpr StringLiteral kEventTypeKey = "event_type";
  static constexpr StringLiteral kCpuSecondsKey = "cpu_seconds";
  static constexpr StringLiteral kCloudTypeKey = "cloud";
  static constexpr StringLiteral kRegionTypeKey = "region";
  static constexpr StringLiteral kInstanceTypeKey = "instance_type";
  static constexpr StringLiteral kInstanceClassKey = "instance_class";

  struct Options {
    // Interval between metering requests. Defaults to 30m.
    DurationType interval = std::chrono::minutes(30);
  };

  /// Generic compute instance identifiers.
  struct InstanceInfo {
    /// Cloud platform instance is running on.
    std::string cloud;

    /// Region instance is located in (e.g. us-west-1)
    std::string region;

    /// Type of instance.
    std::string type;
  };

  static std::unique_ptr<MeteringContext>
  create(MeteringContext::Options options, HTTPContextRef httpCtx,
         size_t maxProcessors);

  static ErrorOrSuccess resolveHostInfo(HostMachineInfo &hostInfo,
                                        InstanceInfo &instInfo);

  MeteringContext(Options o, InstanceInfo ii, size_t mp)
      : interval(o.interval), options(o), instInfo(std::move(ii)),
        maxProcessors(mp), meterCallback({}) {}
  MeteringContext(MeteringContext &&other)
      : interval(other.interval), options(other.options),
        instInfo(other.instInfo), maxProcessors(other.maxProcessors),
        meterCallback(other.meterCallback) {}
  virtual ~MeteringContext() = default;

  void shutdown() {
    if (meterThread.has_value())
      stopMeterThread();
  }

  // Mutually exclusive callback setters.
  ErrorOrSuccess
  setDefaultCallback(M::Telemetry::TelemetryContext &telemetryCtx);
  void setMeterCallback(MeterCallbackFn fn) { meterCallback = std::move(fn); }

  /// Call to send the initial 0-valued metering.
  ErrorOrSuccess start();

  /// Call to send a usage log immediately.
  ErrorOrSuccess flush();

  /// Background meter thread methods.
  void startMeterThread();
  void stopMeterThread();

  TimePoint getLastMeterTime() const { return lastMeterTime.load(); }

private:
  AlignedAtomic<TimePoint> lastMeterTime{ClockType::now()};
  const DurationType interval;

  const Options options;
  const InstanceInfo instInfo;
  const size_t maxProcessors; // May be less than host physical limit.

  MeterCallbackFn meterCallback;

  std::optional<std::thread> meterThread;
  // Unused if the background meter thread has not been started.
  std::condition_variable meterCv;
  std::mutex meterMu;
  bool stopped{false};

  MeteringContext::MeterAttributes getMeterAttributes() const;
  ErrorOrSuccess invokeMeterCallback(DurationType elapsed, bool stopped) const;
};

} // namespace M::Metering

#endif // SUPPORT_METERING_METERINGCONTEXT_H
