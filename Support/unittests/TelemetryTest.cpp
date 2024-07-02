//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Telemetry/Telemetry.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Telemetry/Logs.h"
#include "llvm/Support/MemoryBuffer.h"

#include <thread>

#include <stdlib.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;
using namespace Telemetry;

#ifdef MODULAR_ENABLE_TELEMETRY

/// RAII-style way to restore Modular config after each test.
struct LogFileSetup {
public:
  LogFileSetup(StringRef signalType) {
    EXPECT_THAT(signalType.str(), testing::AnyOf("metrics", "logs"));

    filePathKey = ("telemetry.exporters." + signalType + ".file_path").str();
    httpUrlKey = ("telemetry.exporters." + signalType + ".http_endpoint").str();

    cfg.setEnvOverride(false);
  }

  /// NOTE: This config stays in-memory and is passed to the TelemetryContext
  /// constructor. This is done to isolate the telemetry config used by these
  /// unit tests from Modular's centralized config. There might be other
  /// processes running at the same time as these tests and emitting telemetry,
  /// and we don't want them to write to the same files.
  Config getConfig() { return std::move(cfg); }

  TempFile getLogFile(StringRef prefix, StringRef level) {
    EXPECT_THAT(level.str(), testing::AnyOf("0", "1", "2"));
    auto tmpOr =
        TempFile::create((prefix + "-test-telemetry-%%%%%%%.log").str());
    EXPECT_FALSE(tmpOr.isError()) << tmpOr.getError();

    // Set the config value.
    cfg.setValue(filePathKey, tmpOr->getPath().string());
    // These tests don't need to send to any HTTP endpoint.
    cfg.setValue(httpUrlKey, "");

    // Set telemetry level.
    cfg.setValue("telemetry.level", level);

    // Return the temp file so it'll automatically get destroyed.
    return std::move(*tmpOr);
  }

private:
  Config cfg;
  std::string filePathKey;
  std::string httpUrlKey;
  std::string filePathOriginalValue;
  std::string httpUrlOriginalValue;
};

/// This function parses an OTel message, and provides visitor-style access to
/// the fields in a message. The message looks like this:
///
///  {
///    scope name	: modular
///    schema url	:
///    version	:
///    start time	: Wed Jul 26 21:48:59 2023
///    end time	: Wed Jul 26 21:48:59 2023
///    instrument name	: basic.counter
///    description	:
///    unit		:
///    type		: SumPointData
///    value		: 32
///    attributes		:
///    resources	:
///  	arch: apple-m1
///  	cores: 10
///  	cpu: Apple M1 Max\0
///  	features: []
///  	operating system: darwin
///  	service.name: unknown_service
///  	telemetry.sdk.language: cpp
///  	telemetry.sdk.name: opentelemetry
///  	telemetry.sdk.version: 1.9.1
///  }
///
static void iterateFields(StringRef message,
                          function_ref<void(StringRef, StringRef)> callback) {
  StringRef line;
  while (!message.empty()) {
    line = message.take_until([](char c) { return c == '\n' || c == '}'; });
    if (line.empty())
      return;
    StringRef key, value;
    std::tie(key, value) = line.split(':');
    callback(key.trim(), value.trim());
    // Drop the line we just handled.
    message = message.drop_front(line.size());
    message.consume_front("\n");
  }
}

/// This function provides visitor-style access to all the fields of every
/// message. The OTel text stream looks like this (many new-line delimited
/// messages):
///
/// {
///   scope name: modular
///   ...
/// }
/// {
///   scope name: modular
///   ...
/// }
/// ...
///
/// See above for an example message.
static void iterateMessages(StringRef log,
                            function_ref<void(StringRef, StringRef)> callback) {
  StringRef message;
  while (!log.empty()) {
    log.consume_front("{\n");
    message = log.take_until([](char c) { return c == '}'; });
    if (message == "}")
      return;
    iterateFields(message, callback);
    log = log.drop_front(message.size());
    log.consume_front("}\n");
  }
}

/// This function provides visitor-style access to every message.
static void iterateMessages(StringRef log,
                            function_ref<void(StringRef)> callback) {
  StringRef message;
  while (!log.empty()) {
    log.consume_front("{\n");
    message = log.take_until([](char c) { return c == '}'; });
    if (message == "}")
      return;
    callback(message);
    log = log.drop_front(message.size());
    log.consume_front("}\n");
  }
}

/// This test ensures that when we create and increment a counter, we get the
/// values we expect in the log file, in the order we expect.
TEST(Telemetry, Counter) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("counter", "0");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  auto counter = ctx.createUInt64Counter("basic.test.counter", Level::L0);
  counter.add(32);
  ctx.flush();
  counter.add(10);
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool found32 = false;
        bool foundPlus10 = false;
        StringRef currentInstrument = "";
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          int i;
          if (key == "instrument name") {
            currentInstrument = value;
            return;
          }
          // consumeInteger returns *false* on success.
          if (key == "value" && currentInstrument == "basic.test.counter" &&
              !value.consumeInteger(10, i)) {
            if (i == 32) {
              EXPECT_FALSE(foundPlus10) << "expected to find 32 first";
              found32 = true;
            } else if (i == 42) {
              EXPECT_TRUE(found32)
                  << "expected to find 32 first, found 32 + 10 first?";
              foundPlus10 = true;
            }
          }
        });

        EXPECT_TRUE(found32 && foundPlus10)
            << "expected to find both counter values";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that if we create a histogram and add some records, we get
/// the values we expect in the log file.
/// FIXME(SVCS-218): This test is flaky.
TEST(Telemetry, DISABLED_Histogram) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("histogram", "1");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  std::string value = "ATTRIBUTE";
  llvm::StringMap<MetricAttributeValue> attributes = {{"TELEMETRY", value}};
  auto hist =
      ctx.createUInt64Histogram("basic.test.histogram", Level::L0, attributes);
  hist.record(32);
  hist.record(10);

  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool instrumentFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto instrumentPos = message.find("instrument name");
          StringRef instrumentLine = getLineStartingAt(instrumentPos);
          if (instrumentLine.split(':').second.trim() != "basic.test.histogram")
            return;

          instrumentFound = true;

          auto countPos = message.find("count");
          StringRef countLine = getLineStartingAt(countPos);
          EXPECT_EQ(countLine.split(':').second.trim(), "2");

          auto minPos = message.find("min");
          StringRef minLine = getLineStartingAt(minPos);
          EXPECT_EQ(minLine.split(':').second.trim(), "10");

          auto maxPos = message.find("max");
          StringRef maxLine = getLineStartingAt(maxPos);
          EXPECT_EQ(maxLine.split(':').second.trim(), "32");

          auto bucketsPos = message.find("buckets");
          StringRef bucketsLine = getLineStartingAt(bucketsPos);
          EXPECT_EQ(bucketsLine.split(':').second.trim(),
                    "[0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, "
                    "5000, 7500, 10000, ]");

          auto countsPos = message.find("counts");
          StringRef countsLine = getLineStartingAt(countsPos);
          EXPECT_EQ(countsLine.split(':').second.trim(),
                    "[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]");
        });

        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "TELEMETRY") {
            EXPECT_EQ(value, "ATTRIBUTE");
          }
        });

        EXPECT_TRUE(instrumentFound) << "expected to find histogram in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that measurements for L1 instruments are not emitted
/// when the telemetry level is L0.
TEST(Telemetry, HistogramL1) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("histogram", "0");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  auto hist = ctx.createUInt64Histogram("optional.histogram", Level::L1);
  hist.record(32);
  hist.record(10);
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool instrumentFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "instrument name" && value == "optional.histogram")
            instrumentFound = true;
        });

        EXPECT_FALSE(instrumentFound)
            << "expected not to find histogram in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

// This check tests that our Timer object works, and properly tags attributes
// when it goes out of scope
TEST(Telemetry, Timer) {
  LogFileSetup logFileSetup("metrics");
  TempFile tmpFile = logFileSetup.getLogFile("timer", "0");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  auto lambda_test = [&]() {
    std::string value = "ATTRIBUTE";
    llvm::StringMap<MetricAttributeValue> attrs = {{"TELEMETRY", value}};
    auto timer = ctx.createUInt64Timer<std::chrono::milliseconds>(
        "basic.test.timer", Level::L0, attrs);
    value[0] = 'C';
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };

  lambda_test();
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool instrumentFound = false;
        std::cerr << mbuf->getBuffer().str() << "\n";
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto instrumentPos = message.find("instrument name");
          StringRef instrumentLine = getLineStartingAt(instrumentPos);
          if (instrumentLine.split(':').second.trim() != "basic.test.timer")
            return;

          instrumentFound = true;

          auto maxPos = message.find("max");
          StringRef maxLine = getLineStartingAt(maxPos);
          EXPECT_NE(maxLine.split(':').second.trim().compare_numeric("100"),
                    -1);
        });

        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "TELEMETRY") {
            EXPECT_EQ(value, "ATTRIBUTE");
          }
        });

        EXPECT_TRUE(instrumentFound) << "expected to find timer in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that logs are properly flushed to the log file, escapes and
/// all.
TEST(Telemetry, Logger) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL1Event("test.Logger", {{"attr1", "hello"}, {"attr2", "world"}});
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto eventNamePos = message.find("event.name");
          StringRef eventNameLine = getLineStartingAt(eventNamePos);
          if (eventNameLine.split(':').second.trim() != "test.Logger")
            return;

          eventFound = true;

          // There should be no body
          auto bodyPos = message.find("body");
          StringRef bodyLine = getLineStartingAt(bodyPos);
          EXPECT_EQ(bodyLine.split(':').second.trim(), "");

          auto severityPos = message.find("severity_text");
          StringRef severityLine = getLineStartingAt(severityPos);
          EXPECT_EQ(severityLine.split(':').second.trim(), "INFO");

          auto attribute1Pos = message.find("attr1");
          StringRef attr1Line = getLineStartingAt(attribute1Pos);
          EXPECT_EQ(attr1Line.split(':').second.trim(), "hello");

          auto attribute2Pos = message.find("attr2");
          StringRef attr2Line = getLineStartingAt(attribute2Pos);
          EXPECT_EQ(attr2Line.split(':').second.trim(), "world");
        });

        EXPECT_TRUE(eventFound) << "expected to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that L2 events are not emitted when telemetry
/// level is L1.
TEST(Telemetry, LoggerL2) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));
  TelemetryContext ctx(settings);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL2Event("test.LoggerL2");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          if (key == "event.name" && value == "test.LoggerL2")
            eventFound = true;
        });

        EXPECT_FALSE(eventFound) << "expected not to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

TEST(Telemetry, Resources) {
  LogFileSetup logFileSetup("logs");
  TempFile tmpFile = logFileSetup.getLogFile("log", "1");
  Settings settings(logFileSetup.getConfig(),
                    EntitlementStore::alwaysOpen(llvm::errs()));

  llvm::StringMap<Telemetry::TelemetryContext::AttributeValue> extras;
  StringRef resourceVal = "aResource value here";
  extras["aResource"] = resourceVal;
  extras["aNumber"] = 32;
  TelemetryContext ctx(settings, extras);

  auto logger = ctx.getLogger("basic.log");
  logger->emitL0Event("test.Resources");
  ctx.flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        auto getLineStartingAt = [&](auto pos) {
          StringRef str = mbuf->getBuffer().substr(pos);
          return str.take_until([](char c) { return c == '\n'; });
        };

        bool eventFound = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef message) {
          auto eventNamePos = message.find("event.name");
          StringRef eventNameLine = getLineStartingAt(eventNamePos);
          if (eventNameLine.split(':').second.trim() != "test.Resources")
            return;

          eventFound = true;

          auto resourcePos = message.find("aResource");
          StringRef resourceLine = getLineStartingAt(resourcePos);
          EXPECT_EQ(resourceLine.split(':').second.trim(), resourceVal);

          auto numberPos = message.find("aNumber");
          StringRef numberLine = getLineStartingAt(numberPos);
          EXPECT_EQ(numberLine.split(':').second.trim(), "32");
        });

        EXPECT_TRUE(eventFound) << "expected to find event in file";
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

TEST(Telemetry, LocalIDs) {
  auto origIDs = createLocalIDs();
  EXPECT_EXIT(
      {
        auto newIDs = createLocalIDs();
        // The first ID should be machine invariant.
        if (origIDs.first != newIDs.first)
          exit(1);
        // The second one should be process invariant.
        if (origIDs.second == newIDs.second)
          exit(1);
        exit(0); // Success.
      },
      testing::ExitedWithCode(0), "");
}
#endif // MODULAR_ENABLE_TELEMETRY
