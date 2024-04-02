//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metering/MeteringContext.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"

// AWS
#include "Support/Metering/AWS/InstanceIdentifier.h"

#define DEBUG_TYPE "modular-metering"

namespace M::Metering {

namespace {

void addInstanceFields(const MeteringContext::InstanceInfo &info,
                       MeteringContext::MeterAttributes &attrs) {
  StringRef instance_type(info.type);
  attrs["instance.type"] = instance_type;

  auto found = info.type.find(".");
  attrs["instance.class"] =
      found != std::string::npos ? instance_type.take_front(found) : "";
}

MeteringContext::InstanceInfo resolveInstanceInfo(HTTPContextRef httpCtx) {
  MeteringContext::InstanceInfo info;
  // AWS
  {
    InstanceIdentifier identifier{std::move(httpCtx)};
    if (auto result = identifier.fetch(); result.isError()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to identify AWS instance with error: "
                              << result.getError() << "\n");
    } else {
      info.cloud = "aws";
      info.region = identifier.getRegion();
      info.type = identifier.getInstanceType();
    }
  }
  return info;
}

std::string prefixedName(StringRef name) {
  return (MeteringContext::kEventDomain + "." + name).str();
}

} // namespace

std::unique_ptr<MeteringContext>
MeteringContext::create(MeteringContext::Options options,
                        HTTPContextRef httpCtx, size_t maxProcessors) {
  InstanceInfo info = resolveInstanceInfo(std::move(httpCtx));
  return std::make_unique<MeteringContext>(std::move(options), std::move(info),
                                           maxProcessors);
}

void MeteringContext::setLogCallback(
    M::Telemetry::TelemetryContext &telemetryCtx) {
  setMeterCallback(
      [logger = telemetryCtx.getLogger(MeteringContext::kEventDomain),
       fixed = getMeterAttributes()](int cpuSeconds, bool stopped) {
        auto attrs = fixed;
        attrs[prefixedName("cpu_seconds")] = cpuSeconds;
        logger->emitL0Event(MeteringContext::kEventName, attrs);
        return success();
      });
}

ErrorOrSuccess MeteringContext::flush() {
  const auto now = ClockType::now();
  const auto last = lastMeterTime.load();
  const auto duration = now - last;
  // TODO: Buffer batch usage (for up to 6 hours in the past) if network dies.
  const auto seconds =
      std::chrono::duration_cast<
          std::chrono::duration<double, std::chrono::seconds::period>>(duration)
          .count();
  if (meterCallback) {
    auto outcome = meterCallback(maxProcessors * seconds, stopped);
    if (outcome.isError())
      return Error("Received error from meter callback: " +
                   llvm::StringRef(outcome.getError()));
  }
  lastMeterTime = now;
  return success();
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
  MeteringContext::MeterAttributes attrs = {
      {"event_type", MeteringContext::kEventType},
  };
  // Instance info
  {
    attrs["cloud"] = instInfo.cloud;
    attrs["region"] = instInfo.region;
    addInstanceFields(instInfo, attrs);
  }
  return attrs;
}

} // namespace M::Metering
