//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Billing/BillingContext.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Support/Threading/HWInfo.h"

namespace M::Billing {

using ::testing::Unused;

class BillingTest : public ::testing::Test {
protected:
  void createContext(const BillingContext::Options &options,
                     bool meterError = false) {
    context = std::make_unique<BillingContext>(
        options, BillingContext::InstanceInfo{"", ""},
        [&, err = meterError](int millis) -> ErrorOrSuccess {
          values.push_back(millis);
          if (err)
            return Error("error");
          return success();
        });
  }

  std::unique_ptr<BillingContext> context;
  std::vector<int> values;
};

constexpr static double kEps = 10;

TEST_F(BillingTest, FlushSuccess) {
  createContext({});

  auto errOr = context->flush();
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  ASSERT_EQ(values.size(), 1u);
  const auto numCores = getNumPhysicalCores();
  EXPECT_NEAR(values[0], numCores * 1 * 3600, kEps);
}

TEST_F(BillingTest, FlushFractional) {
  createContext({});

  // Fractional flush will pro-rate.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto errOr = context->flush(true);
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  ASSERT_EQ(values.size(), 1u);
  const auto numCores = getNumPhysicalCores();
  const auto elapsedHrs =
      std::chrono::duration_cast<
          std::chrono::duration<double, std::chrono::hours::period>>(
          std::chrono::steady_clock::now() - context->getStart())
          .count();
  EXPECT_NEAR(values.back(), numCores * elapsedHrs * 3600, kEps);
}

TEST_F(BillingTest, FlushFractionalWithThread) {
  ;
  BillingContext::Options opts;
  opts.intervalMs = 3;
  createContext(opts);

  context->startMeterThread();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  context->stopMeterThread();

  const auto numCores = getNumPhysicalCores();
  const auto elapsedHrs =
      std::chrono::duration_cast<
          std::chrono::duration<double, std::chrono::hours::period>>(
          std::chrono::steady_clock::now() - context->getStart())
          .count();
  EXPECT_NEAR(values.back(), numCores * elapsedHrs * 3600, kEps);
}

TEST_F(BillingTest, FlushFailure) {
  createContext({}, true);
  auto errOr = context->flush();
  ASSERT_TRUE(errOr.isError());
}

TEST_F(BillingTest, FlushOnDestruct) {
  createContext({});
  context.reset();
  ASSERT_FALSE(values.empty());
}

TEST_F(BillingTest, StopMeterThread) {
  createContext({});
  context->startMeterThread();
  // Destructor stops meter thread before 1h.
}

} // namespace M::Billing
