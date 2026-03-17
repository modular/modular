//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Log.h"

#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>

#ifndef _WIN32
#include <stdlib.h>
#endif

#include "gtest/gtest.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace M::Log;

namespace {

static std::string gLogFilePath;
static std::string gConfigDirRoot;

class LogJSONOutputTestEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    // Create a unique temp directory: /tmp/modular-log-json-test-XXXXXX
    llvm::SmallString<128> tmpDir;
    auto ec =
        llvm::sys::fs::createUniqueDirectory("modular-log-json-test", tmpDir);
    ASSERT_FALSE(ec) << "Failed to create temp config dir: " << ec.message();
    gConfigDirRoot = tmpDir.str().str();

    // Create /tmp/<random>/.modular/ subdirectory
    llvm::SmallString<128> configDir = tmpDir;
    llvm::sys::path::append(configDir, ".modular");
    ec = llvm::sys::fs::create_directory(configDir);
    ASSERT_FALSE(ec) << "Failed to create .modular dir: " << ec.message();

    // Set TEST_TMPDIR so Config::open() searches <tmpDir>/.modular for config.
#ifndef _WIN32
    setenv("TEST_TMPDIR", tmpDir.c_str(), /*overwrite=*/1);
#else
#error "Test requires modification for Windows"
#endif

    // Create a unique temp file for log output.
    char tmpPath[] = "/tmp/modular-log-json-test-%%%%%%";
    int fd = 0;
    llvm::SmallString<128> realPath;
    ec = llvm::sys::fs::createUniqueFile(tmpPath, fd, realPath);
    ASSERT_FALSE(ec) << "Failed to create temp log file: " << ec.message();
    gLogFilePath = realPath.str().str();

    // Build and write modular.cfg into the .modular directory.
    llvm::SmallString<128> configFilePath = configDir;
    llvm::sys::path::append(configFilePath, "modular.cfg");

    llvm::SmallString<256> configContent;
    configContent += R"(
[log]
file = )";
    configContent += gLogFilePath;
    configContent += "\n";

    llvm::raw_fd_ostream cfgFile(configFilePath, ec);
    ASSERT_FALSE(ec) << "Failed to open modular.cfg for writing: "
                     << ec.message();
    cfgFile << configContent;
  }

  void TearDown() override {
    llvm::sys::fs::remove(gLogFilePath);
    llvm::sys::fs::remove_directories(gConfigDirRoot);
  }
};

static ::testing::Environment *const kLogEnv =
    ::testing::AddGlobalTestEnvironment(new LogJSONOutputTestEnvironment);

std::streampos currentLogEnd() {
  std::ifstream f(gLogFilePath);
  f.seekg(0, std::ios::end);
  return f.tellg();
}

std::string readLogSince(std::streampos offset) {
  std::ifstream f(gLogFilePath);
  f.seekg(offset);
  return {std::istreambuf_iterator<char>(f), {}};
}

class LogJSONTest : public ::testing::Test {
protected:
  void SetUp() override { startPos_ = currentLogEnd(); }

  std::string capturedOutput() const { return readLogSince(startPos_); }

private:
  std::streampos startPos_{};
};

TEST_F(LogJSONTest, OutputIsOnOneLine) {
  MLOG(LogLevel::INFO, "single line");
  auto out = capturedOutput();
  // Exactly one newline, at the end.
  EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 1);
}

TEST_F(LogJSONTest, TimestampIsISO8601WithMicroseconds) {
  MLOG(LogLevel::INFO, "ts check");
  std::regex tsPattern(
      R"("timestamp"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")");
  EXPECT_TRUE(std::regex_search(capturedOutput(), tsPattern));
}

TEST_F(LogJSONTest, LevelFieldDebug) {
  MLOG(LogLevel::DEBUG, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"DBG")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldInfo) {
  MLOG(LogLevel::INFO, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"INFO")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldWarn) {
  MLOG(LogLevel::WARN, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"WARN")"), std::string::npos);
}

TEST_F(LogJSONTest, LevelFieldError) {
  MLOG(LogLevel::ERROR, "msg");
  EXPECT_NE(capturedOutput().find(R"("level":"ERR")"), std::string::npos);
}

TEST_F(LogJSONTest, MessageFieldIsPresent) {
  MLOG(LogLevel::INFO, "hello world");
  EXPECT_NE(capturedOutput().find(R"("message":"hello world")"),
            std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesDoubleQuote) {
  MLOG(LogLevel::INFO, R"(say "hi")");
  EXPECT_NE(capturedOutput().find(R"("message":"say \"hi\"")"),
            std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesBackslash) {
  MLOG(LogLevel::INFO, R"(a\b)");
  EXPECT_NE(capturedOutput().find(R"("message":"a\\b")"), std::string::npos);
}

TEST_F(LogJSONTest, MessageEscapesNewline) {
  MLOG(LogLevel::INFO, "line1\nline2");
  EXPECT_NE(capturedOutput().find(R"("message":"line1\nline2")"),
            std::string::npos);
}

TEST_F(LogJSONTest, NoANSIColorCodes) {
  MLOG(LogLevel::WARN, "colorless");
  // ESC character should not appear anywhere in the output.
  EXPECT_EQ(capturedOutput().find('\x1b'), std::string::npos);
}

TEST_F(LogJSONTest, LevelFilteringSuppressesOutput) {
  auto level = getLogLevel();
  setLogLevel(LogLevel::ERROR);
  MLOG(LogLevel::DEBUG, "should be suppressed");
  MLOG(LogLevel::INFO, "also suppressed");
  MLOG(LogLevel::WARN, "also suppressed");
  EXPECT_TRUE(capturedOutput().empty());
  setLogLevel(level);
}

} // namespace
