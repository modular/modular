//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Telemetry/Telemetry.h"
#include "Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

using namespace M;
using namespace LLCL;
using namespace Telemetry;

#ifdef MODULAR_ENABLE_TELEMETRY

static TempFile setupLogFile(StringRef prefix) {
  auto cfgOr = Config::open();
  EXPECT_FALSE(cfgOr.isError()) << cfgOr.getError();

  auto tmpOr = TempFile::create((prefix + "-test-telemetry-%%%%%%%.log").str());
  EXPECT_FALSE(tmpOr.isError()) << tmpOr.getError();

  // Set the config value.
  cfgOr->setValue("telemetry.exporters.metrics.file_path",
                  tmpOr->getPath().string());

  // Flush the config to the file so we can read it from the context.
  auto err = cfgOr->flush();
  EXPECT_FALSE(err.isError()) << err.getError();

  // Return the temp file so it'll automatically get destroyed.
  return std::move(*tmpOr);
}

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

/// This test ensures that when we create and increment a counter, we get the
/// values we expect in the log file, in the order we expect.
TEST(Telemetry, Counter) {
  TempFile tmpFile = setupLogFile("counter");

  RCRef<TelemetryContext> ctx = RCRef<TelemetryContext>::create();

  auto counter = ctx->createUInt64Counter("basic.counter");
  counter.add(32);
  ctx->flush();
  counter.add(10);
  ctx->flush();

  auto err = readFileUnderLock(
      tmpFile.getPath(), [&](const std::filesystem::path &path) {
        auto mbufOr = llvm::MemoryBuffer::getFile(path.string(),
                                                  /*IsText=*/true);
        EXPECT_TRUE(mbufOr) << mbufOr.getError().message();
        std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

        bool found32 = false;
        bool foundPlus10 = false;
        iterateMessages(mbuf->getBuffer(), [&](StringRef key, StringRef value) {
          int i;
          // consumeInteger returns *false* on success.
          if (key == "value" && !value.consumeInteger(10, i)) {
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
TEST(Telemetry, Histogram) {
  TempFile tmpFile = setupLogFile("histogram");

  RCRef<TelemetryContext> ctx = RCRef<TelemetryContext>::create();

  auto hist = ctx->createUInt64Histogram("basic.histogram");
  hist.record(32);
  hist.record(10);
  ctx->flush();

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

        auto countPos = mbuf->getBuffer().find("count");
        StringRef countLine = getLineStartingAt(countPos);
        EXPECT_EQ(countLine.split(':').second.trim(), "2");

        auto minPos = mbuf->getBuffer().find("min");
        StringRef minLine = getLineStartingAt(minPos);
        EXPECT_EQ(minLine.split(':').second.trim(), "10");

        auto maxPos = mbuf->getBuffer().find("max");
        StringRef maxLine = getLineStartingAt(maxPos);
        EXPECT_EQ(maxLine.split(':').second.trim(), "32");

        auto bucketsPos = mbuf->getBuffer().find("buckets");
        StringRef bucketsLine = getLineStartingAt(bucketsPos);
        EXPECT_EQ(bucketsLine.split(':').second.trim(),
                  "[0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, "
                  "5000, 7500, 10000, ]");

        auto countsPos = mbuf->getBuffer().find("counts");
        StringRef countsLine = getLineStartingAt(countsPos);
        EXPECT_EQ(countsLine.split(':').second.trim(),
                  "[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]");
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}

/// This test checks that logs are properly flushed to the log file, escapes and
/// all.
TEST(Telemetry, Logger) {
  TempFile tmpFile = setupLogFile("log");

  RCRef<TelemetryContext> ctx = RCRef<TelemetryContext>::create();

  StringRef logString = "hello\nthis is a string";
  StringRef escapedLogString = "hello\\nthis is a string";

  auto logger = ctx->getLogger("basic.log");
  logger->getInfo("test") << logString;
  ctx->flush();

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

        auto bodyPos = mbuf->getBuffer().find("body");
        StringRef bodyLine = getLineStartingAt(bodyPos);
        EXPECT_EQ(bodyLine.split(':').second.trim(), escapedLogString);

        auto severityPos = mbuf->getBuffer().find("severity_text");
        StringRef severityLine = getLineStartingAt(severityPos);
        EXPECT_EQ(severityLine.split(':').second.trim(), "INFO");
      });
  EXPECT_FALSE(err.isError()) << err.getError();
}
#endif // MODULAR_ENABLE_TELEMETRY
