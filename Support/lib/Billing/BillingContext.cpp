//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Billing/BillingContext.h"

#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"

// AWS
#include "Support/Billing/AWS/InstanceIdentifier.h"

#define DEBUG_TYPE "modular-billing"

namespace M::Billing {

ErrorOr<std::unique_ptr<BillingContext>>
BillingContext::createForAWS(BillingContext::Options options,
                             HTTPContextRef httpCtx,
                             std::function<ErrorOrSuccess(int)> meterCallback) {
  InstanceIdentifier identifier{httpCtx.copy()};
  InstanceInfo info;
  if (auto result = identifier.fetch(); result.isError()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to identify instance with error: "
                            << result.getError() << "\n");
  } else {
    info.region = identifier.getRegion();
    info.type = identifier.getInstanceType();
  }
  return std::make_unique<BillingContext>(std::move(options), std::move(info),
                                          std::move(meterCallback));
}

BillingContext::BillingContext(Options o, InstanceInfo i,
                               std::function<ErrorOrSuccess(int)> fn)
    : options(std::move(o)), info(std::move(i)), interval(options.intervalMs),
      meterCallback(std::move(fn)) {}

BillingContext::~BillingContext() {
  if (meterThread.has_value())
    stopMeterThread();
  else
    (void)flush(true);
}

ErrorOrSuccess BillingContext::flush(bool fractional) {
  const auto now = ClockType::now();
  const auto start = startTime.load();
  const auto duration = fractional ? now - start : interval;
  // TODO: Buffer batch usage (for up to 6 hours in the past) if network dies.
  const auto seconds =
      std::chrono::duration_cast<
          std::chrono::duration<double, std::chrono::seconds::period>>(duration)
          .count();
  auto maxProcessors = getNumPhysicalCores();
  auto limitsOr = CPULimits::get();
  if (!limitsOr.isError()) {
    auto millicores = limitsOr->millicores;
    if (millicores.has_value())
      maxProcessors = *millicores / 1000;
  }

  auto outcome = meterCallback(maxProcessors * seconds);
  if (outcome.isError())
    return Error("Received error from meter callback: " +
                 llvm::StringRef(outcome.getError()));
  return success();
}

void BillingContext::startMeterThread() {
  meterThread.emplace([=]() {
    llvm::set_thread_name("Billing Background Thread");
    std::unique_lock<std::mutex> lock(meterMu);
    while (true) {
      meterCv.wait_until(lock, getStart() + interval,
                         [=]() { return stopped; });
      if (auto errOr = flush(stopped))
        LLVM_DEBUG(llvm::dbgs() << "Error sending meter request: "
                                << errOr.getError() << "\n");
      if (stopped)
        return;

      advanceStart();
    }
  });
}

void BillingContext::stopMeterThread() {
  if (meterThread.has_value() && meterThread->joinable()) {
    {
      std::lock_guard<std::mutex> lock(meterMu);
      stopped = true;
    }
    meterCv.notify_one();
    meterThread->join();
  }
}

} // namespace M::Billing
