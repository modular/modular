//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metering/MeteringContext.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"
#include <iostream>

// AWS
#include "Support/Metering/AWS/InstanceIdentifier.h"

#define DEBUG_TYPE "modular-metering"

namespace M::Metering {

namespace {

void addInstanceFields(const MeteringContext::InstanceInfo &info,
                       MeteringContext::MeterAttributes &attrs) {
  if (info.type.empty())
    return;

  StringRef instance_type(info.type);
  attrs[MeteringContext::kInstanceTypeKey] = instance_type;

  auto dot_found = info.type.find(".");
  attrs[MeteringContext::kInstanceClassKey] =
      dot_found != std::string::npos ? instance_type.take_front(dot_found) : "";
}

constexpr StringLiteral kCloudAws = "aws";

MeteringContext::InstanceInfo resolveInstanceInfo(HTTPContextRef httpCtx) {
  MeteringContext::InstanceInfo info;
  // AWS
  {
    InstanceIdentifier identifier{std::move(httpCtx)};
    if (auto result = identifier.fetch(); result.isError()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to identify AWS instance with error: "
                              << result.getError() << "\n");
    } else {
      info.cloud = kCloudAws;
      info.region = identifier.getRegion();
      info.type = identifier.getInstanceType();
    }
  }
  return info;
}

} // namespace

std::unique_ptr<MeteringContext>
MeteringContext::create(MeteringContext::Options options,
                        HTTPContextRef httpCtx, size_t maxProcessors) {
  InstanceInfo info = resolveInstanceInfo(std::move(httpCtx));
  return std::make_unique<MeteringContext>(options, std::move(info),
                                           maxProcessors);
}

ErrorOrSuccess MeteringContext::setDefaultCallback(
    M::Telemetry::TelemetryContext &telemetryCtx) {
  auto logger = telemetryCtx.getLogger(MeteringContext::kEventDomain);
  setMeterCallback([this, logger = std::move(logger),
                    fixed = getMeterAttributes()](DurationType duration,
                                                  bool stopped) {
    auto attrs = fixed;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    attrs[kCpuSecondsKey] = static_cast<int>(seconds.count() * maxProcessors);
    logger->emitL0Event(MeteringContext::kEventName, attrs);
    return success();
  });
  // Send initial 0-valued data point.
  return start();
}

ErrorOrSuccess MeteringContext::start() {
  return invokeMeterCallback(std::chrono::seconds(1), stopped);
}

ErrorOrSuccess MeteringContext::flush() {
  const auto now = ClockType::now();
  const auto last = lastMeterTime.load();
  // TODO: Buffer batch usage (for up to 6 hours in the past) if network dies.
  const auto elapsed = std::chrono::ceil<std::chrono::seconds>(
      std::chrono::duration_cast<std::chrono::seconds>(now - last));
  lastMeterTime.store(now);
  return invokeMeterCallback(elapsed, stopped);
}

void MeteringContext::startMeterThread() {
  if (meterThread.has_value())
    // Already started.
    return;

  meterThread.emplace([=]() {
    llvm::set_thread_name("Metering Background Thread");
    std::unique_lock<std::mutex> lock(meterMu);
    while (true) {
      meterCv.wait_until(lock, getLastMeterTime() + interval,
                         [=]() { return stopped; });
      if (auto errOr = flush())
        LLVM_DEBUG(llvm::dbgs() << "Error sending meter request: "
                                << errOr.getError() << "\n");
      if (stopped)
        return;
    }
  });
}

void MeteringContext::stopMeterThread() {
  if (meterThread.has_value() && meterThread->joinable()) {
    {
      std::lock_guard<std::mutex> lock(meterMu);
      stopped = true;
    }
    meterCv.notify_one();
    meterThread->join();
  }
}

MeteringContext::MeterAttributes MeteringContext::getMeterAttributes() const {
  MeteringContext::MeterAttributes attrs = {{kEventTypeKey, kEventType}};
  // Instance info
  {
    if (!instInfo.cloud.empty())
      attrs[kCloudTypeKey] = instInfo.cloud;
    if (!instInfo.region.empty())
      attrs[kRegionTypeKey] = instInfo.region;
    addInstanceFields(instInfo, attrs);
  }
  return attrs;
}

ErrorOrSuccess MeteringContext::invokeMeterCallback(DurationType elapsed,
                                                    bool stopped) const {
  if (meterCallback) {
    auto outcome = meterCallback(elapsed, stopped);
    if (outcome.isError())
      return Error("Received error from meter callback: " +
                   llvm::StringRef(outcome.getError()));
  }
  return success();
}

} // namespace M::Metering
