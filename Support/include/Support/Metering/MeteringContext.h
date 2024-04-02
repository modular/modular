//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METERING_BILLINGCONTEXT_H
#define SUPPORT_METERING_BILLINGCONTEXT_H

#include <chrono>
#include <condition_variable>
#include <thread>

#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/MArchTarget/Host.h"
#include "Support/Telemetry/Telemetry.h"

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
  using Duration = std::chrono::duration<double>;

  /// Called every interval with elapsed CPU seconds + shutdown status.
  using MeterCallbackFn = std::function<ErrorOrSuccess(int, bool)>;
  using MeterAttributes = llvm::StringMap<M::Telemetry::Logs::AttributeValue>;

  static constexpr llvm::StringRef kEventDomain = "metering";
  static constexpr llvm::StringRef kEventName = "meter";
  static constexpr llvm::StringRef kEventType = "cpu_usage_v1";

  struct Options {
    // Logs to stdout if true, otherwise uploads.
    bool dryRun{true};

    // Times to retry registration if throttled.
    size_t retryCount{4};

    // Interval between metering requests. Defaults to 1h.
    size_t intervalMs{60 * 60 * 1000};
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
      : options(std::move(o)), instInfo(std::move(ii)), maxProcessors(mp),
        interval(options.intervalMs), meterCallback({}) {}
  MeteringContext(MeteringContext &&other)
      : options(other.options), instInfo(other.instInfo),
        maxProcessors(other.maxProcessors), interval(other.interval),
        meterCallback(other.meterCallback) {}
  virtual ~MeteringContext() = default;

  void shutdown() {
    if (meterThread.has_value())
      stopMeterThread();

    (void)flush();
  }

  void setLogCallback(M::Telemetry::TelemetryContext &telemetryCtx);
  void setMeterCallback(MeterCallbackFn fn) { meterCallback = std::move(fn); }

  /// Call to send a usage log immediately.
  ErrorOrSuccess flush();

  /// Background meter thread methods.
  void startMeterThread();
  void stopMeterThread();

  TimePoint getLastMeterTime() const { return lastMeterTime.load(); }

private:
  const Options options;
  const InstanceInfo instInfo;
  const size_t maxProcessors; // May be less than host physical limit.

  const std::chrono::milliseconds interval;
  std::atomic<TimePoint> lastMeterTime{ClockType::now()};

  MeterCallbackFn meterCallback;

  std::optional<std::thread> meterThread;
  // Unused if the background meter thread has not been started.
  std::condition_variable meterCv;
  std::mutex meterMu;
  bool stopped{false};

  MeteringContext::MeterAttributes getMeterAttributes() const;
};

} // namespace M::Metering

#endif // SUPPORT_METERING_BILLINGCONTEXT_H
