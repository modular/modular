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
  ~MeteringContextTest() {
    if (context)
      context->shutdown();
  }

  void createContext(const MeteringContext::Options &options,
                     bool meterError = false) {
    context = std::make_unique<MeteringContext>(
        options, MeteringContext::InstanceInfo{"", "", ""}, 5);
    context->setMeterCallback(
        [&, err = meterError](int millis, bool stopped) -> ErrorOrSuccess {
          if (!stopped) {
            values.push_back(millis);
          }
          if (err)
            return Error("error");
          return success();
        });
  }

  int elapsedSeconds() const {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::chrono::seconds::period>>(
               std::chrono::steady_clock::now() - context->getLastMeterTime())
        .count();
  }

  std::unique_ptr<MeteringContext> context;
  std::vector<int> values;
};

constexpr static double kEps = 10;

TEST_F(MeteringContextTest, FlushSuccess) {
  createContext({});

  auto errOr = context->flush();
  ASSERT_FALSE(errOr.isError()) << errOr.getError();
  ASSERT_EQ(values.size(), 1u);
  const auto numCores = getNumPhysicalCores();
  EXPECT_NEAR(values.back(), numCores * elapsedSeconds(), kEps);
}

TEST_F(MeteringContextTest, FlushSuccessMultiple) {
  createContext({});

  std::this_thread::sleep_for(std::chrono::seconds(1));
  auto errOr = context->flush();
  auto fn = [&] {
    ASSERT_FALSE(errOr.isError()) << errOr.getError();
    const auto numCores = getNumPhysicalCores();
    EXPECT_NEAR(values.back(), numCores * elapsedSeconds(), kEps);
  };
  fn();

  std::this_thread::sleep_for(std::chrono::seconds(1));
  errOr = context->flush();
  fn();

  ASSERT_EQ(values.size(), 2u); // fn() will check each value individually.
}

TEST_F(MeteringContextTest, FlushFailure) {
  createContext({}, true);
  auto errOr = context->flush();
  ASSERT_TRUE(errOr.isError());
}

TEST_F(MeteringContextTest, FlushOnShutdown) {
  createContext({});
  context->shutdown();
  ASSERT_FALSE(values.empty());
}

TEST_F(MeteringContextTest, FlushWithThreadIdempotent) {
  MeteringContext::Options opts;
  opts.intervalMs = 3;

  std::vector<std::pair<std::thread::id, int>> markedValues;
  context = std::make_unique<MeteringContext>(
      std::move(opts), MeteringContext::InstanceInfo{"", "", ""}, 5);
  context->setMeterCallback([&](int millis, bool stopped) -> ErrorOrSuccess {
    if (!stopped) {
      markedValues.emplace_back(std::this_thread::get_id(), millis);
    }
    return success();
  });

  context->startMeterThread();
  context->startMeterThread();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  context->stopMeterThread();

  ASSERT_FALSE(markedValues.empty());
  auto firstThread = markedValues[0].first;
  EXPECT_NE(std::this_thread::get_id(), firstThread);
  EXPECT_THAT(markedValues, Each(Key(firstThread)));
}

TEST_F(MeteringContextTest, StopMeterThread) {
  createContext({});
  context->startMeterThread();
  // Destructor stops meter thread before 1h.
}

} // namespace M::Metering
