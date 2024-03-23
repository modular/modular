//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BILLING_BILLINGCONTEXT_H
#define SUPPORT_BILLING_BILLINGCONTEXT_H

#include <chrono>
#include <thread>

#include "Support/ErrorOr.h"
#include "Support/HTTP/HTTPClient.h"

namespace M::Billing {

/// Governs the calculation and metering of the Modular Compute Unit (MCU) on
/// customer deployments.
///
/// Periodically uploads on a background thread or, if the added thread is
/// undesirable, flush() can be explicitly called.
class BillingContext {
public:
  using ClockType = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<ClockType>;
  using Duration = std::chrono::duration<double>;

  /// Called every configured interval with elapsed CPU seconds.
  using MeterCallbackFn = std::function<ErrorOrSuccess(int)>;

  struct Options {
    // Path to the MCU conversion rate CSV.
    std::string ratesPath;

    // Logs to stdout if true, otherwise uploads.
    bool dryRun{true};

    // Times to retry registration if throttled.
    size_t retryCount{4};

    // Interval between metering requests. Defaults to 1h.
    size_t intervalMs{3600000};
  };

  /// Generic compute instance identifiers.
  struct InstanceInfo {
    /// Region instance is located in (e.g. us-west-1)
    std::string region;

    /// Type of instance.
    std::string type;
  };

  static ErrorOr<std::unique_ptr<BillingContext>>
  createForAWS(BillingContext::Options options, HTTPContextRef httpCtx,
               std::function<ErrorOrSuccess(int)> meterCallback);

  BillingContext(Options o, InstanceInfo i, MeterCallbackFn fn);
  ~BillingContext();

  /// Call to send a usage log immediately.
  ErrorOrSuccess flush(bool fractional = false);

  /// Background meter thread methods.
  void startMeterThread();
  void stopMeterThread();

  /// Start time related methods.
  TimePoint getStart() const { return startTime.load(); }
  void advanceStart() { startTime = getStart() + interval; }
  bool flushedThisHour() const { return ClockType::now() < getStart(); };

private:
  const Options options;
  const InstanceInfo info;
  const std::chrono::milliseconds interval;

  // Start time of current billing hour.
  std::atomic<TimePoint> startTime{ClockType::now()};
  MeterCallbackFn meterCallback;

  std::optional<std::thread> meterThread;
  // Unused if the background meter thread has not been started.
  std::condition_variable meterCv;
  std::mutex meterMu;
  bool stopped{false};
};

} // namespace M::Billing

#endif // SUPPORT_BILLING_BILLINGCONTEXT_H
