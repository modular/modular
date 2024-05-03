//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metering/MeteringContext.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Support/Threading/HWInfo.h"

namespace M::Metering {

using ::testing::Each;
using ::testing::Key;
using ::testing::Unused;

class MeteringContextTest : public ::testing::Test {
protected:
  ~MeteringContextTest() override {
    if (context)
      context->shutdown();
  }

  void createContext(const MeteringContext::Options &options,
                     bool err = false) {
    context = std::make_unique<MeteringContext>(
        options, MeteringContext::InstanceInfo{"", "", ""}, 5);
    expectValuesAdded(err);
    ASSERT_FALSE(context->start().isError());
  }

  MeteringContext::DurationType elapsedSeconds() const {
    return std::chrono::ceil<std::chrono::seconds>(
        std::chrono::steady_clock::now() - context->getLastMeterTime());
  }

  void expectValuesAdded(bool err = false) {
    context->setMeterCallback(
        [=, first = true](MeteringContext::DurationType duration,
                          bool stopped) mutable -> ErrorOrSuccess {
          if (first)
            first = false;
          else if (err)
            return Error("error");

          if (!stopped) {
            {
              std::lock_guard<std::mutex> lk(mu);
              values.emplace_back(std::this_thread::get_id(), duration);
            }
            cv.notify_all();
          }
          return success();
        });
  }

  void waitForValues() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [=] { return !values.empty(); });
  }

  std::unique_ptr<MeteringContext> context;

  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::pair<std::thread::id, MeteringContext::DurationType>> values;
};

constexpr static double kSecondsEps = 2;

TEST_F(MeteringContextTest, FlushSuccess) {
  createContext({});

  auto errOr = context->flush();
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  ASSERT_EQ(values.size(), 2u);
  EXPECT_NEAR(values.back().second.count(), std::chrono::seconds(1).count(),
              kSecondsEps); // Rounds up to 1.
}

TEST_F(MeteringContextTest, FlushSuccessMultiple) {
  createContext({});
  ErrorOrSuccess errOr;

  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  EXPECT_EQ(values.back().second, std::chrono::seconds(1));

  std::this_thread::sleep_for(std::chrono::seconds(1));
  errOr = context->flush();
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  EXPECT_NEAR(values.back().second.count(), elapsedSeconds().count(),
              kSecondsEps);

  std::this_thread::sleep_for(std::chrono::seconds(1));
  errOr = context->flush();
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  EXPECT_NEAR(values.back().second.count(), elapsedSeconds().count(),
              kSecondsEps);
}

TEST_F(MeteringContextTest, FlushFailure) {
  createContext({}, true);
  auto errOr = context->flush();
  ASSERT_TRUE(errOr.isError());
}

TEST_F(MeteringContextTest, FlushOnShutdown) {
  createContext({});
  context->shutdown();
  ASSERT_EQ(values.size(), 2u);
}

TEST_F(MeteringContextTest, FlushWithThreadIdempotent) {
  MeteringContext::Options opts;
  opts.interval = std::chrono::milliseconds(1);

  context = std::make_unique<MeteringContext>(
      opts, MeteringContext::InstanceInfo{"", "", ""}, 5);
  expectValuesAdded();
  context->startMeterThread();
  context->startMeterThread();
  waitForValues();
  context->stopMeterThread();

  ASSERT_FALSE(values.empty());
  auto firstThread = values[0].first;
  EXPECT_NE(std::this_thread::get_id(), firstThread);
  EXPECT_THAT(values, Each(Key(firstThread)));
}

TEST_F(MeteringContextTest, StopMeterThread) {
  createContext({});
  context->startMeterThread();
  // Destructor stops meter thread before 1h.
}

} // namespace M::Metering
