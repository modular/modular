//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metering/MeteringContext.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "Support/Telemetry/Logs.h"
#include "Support/Telemetry/Telemetry.h"
#include "Support/Threading/HWInfo.h"

namespace llvm {

std::string print(const M::Telemetry::Logs::AttributeValue &value) {
  return std::visit(
      [](auto &&arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, const char *> ||
                      std::is_same_v<T, std::string_view>)
          return std::string(arg);
        else if constexpr (std::is_arithmetic_v<T>)
          return std::to_string(arg);
        return "";
      },
      value);
}

void PrintTo(const llvm::StringMap<M::Telemetry::Logs::AttributeValue> &map,
             std::ostream *os) {
  *os << "{ ";
  int i = 0;
  for (auto it = map.begin(); it != map.end(); ++it) {
    if (i > 0)
      *os << ", ";

    *os << std::string(it->getKey()) << ": " << print(it->getValue());
    i++;
  }
  *os << " }";
}
} // namespace llvm

namespace M::Metering {

using namespace M::Telemetry;

using ::testing::_;
using ::testing::NiceMock;

// TODO: Move these somewhere common in Telemetry

class MockEventLogger : public opentelemetry::logs::EventLogger {
public:
  MockEventLogger() = default;

  MOCK_METHOD((const opentelemetry::nostd::string_view), GetName, (),
              (noexcept, override));

  MOCK_METHOD((opentelemetry::nostd::shared_ptr<opentelemetry::logs::Logger>),
              GetDelegateLogger, (), (noexcept, override));

  MOCK_METHOD(
      void, EmitEvent,
      ((opentelemetry::nostd::string_view event_name),
       (opentelemetry::nostd::unique_ptr<opentelemetry::logs::LogRecord> &&
        log_record)),
      (noexcept, override));
};

class MockLogger : public Logs::Logger {
public:
  using Logger::Logger;

  MOCK_METHOD(void, emitL0Event,
              ((llvm::StringRef eventName),
               (const llvm::StringMap<Logs::AttributeValue> &attributes)),
              (override));
  MOCK_METHOD(void, emitL1Event,
              ((llvm::StringRef eventName),
               (const llvm::StringMap<Logs::AttributeValue> &attributes)),
              (override));
  MOCK_METHOD(void, emitL2Event,
              ((llvm::StringRef eventName),
               (const llvm::StringMap<Logs::AttributeValue> &attributes)),
              (override));
  MOCK_METHOD(void, emitL0Error,
              ((llvm::StringRef eventName),
               (const CodedErrorOrSuccess &codedError),
               (const llvm::StringMap<Logs::AttributeValue> &attributes)),
              (override));
};

class MockTelemetryContext : public Telemetry::TelemetryContext {
public:
  MockTelemetryContext(Settings &settings)
      : Telemetry::TelemetryContext(settings) {
    mockEventLogger = std::make_shared<testing::NiceMock<MockEventLogger>>();
    mockLogger = std::make_shared<testing::NiceMock<MockLogger>>(
        mockEventLogger, M::Telemetry::Level::L0);
  }

  std::shared_ptr<Logs::Logger> getLogger(StringRef eventDomain) override {
    return mockLogger;
  }

  std::shared_ptr<testing::NiceMock<MockLogger>> mockLogger;
  std::shared_ptr<testing::NiceMock<MockEventLogger>> mockEventLogger;
};

class MockHTTPClient : public HTTPClient {
public:
  using HTTPClient::HTTPClient;

  MOCK_METHOD(HTTPResponse, executeRequestImpl,
              (const HTTPRequest &request, raw_ostream &os,
               std::chrono::milliseconds timeout, size_t maxLength),
              (override));
};

class MeteringContextLogTest : public ::testing::Test {
protected:
  ~MeteringContextLogTest() override {
    if (context)
      context->shutdown();
  }

  void createContext(const MeteringContext::Options &options,
                     size_t maxProcessors = 8) {
    context = std::make_unique<MeteringContext>(
        options,
        MeteringContext::InstanceInfo{"aws", "us-west-2", "c5.4xlarge"},
        maxProcessors);
    ASSERT_FALSE(context->setDefaultCallback(mockTelemetryCtx).isError());
  }

  static HTTPResponse emptyResponse(const HTTPRequest &request) { return {}; }

  std::unique_ptr<HTTPClient> createHTTPClient(HTTPContextRef ref) {
    auto ptr = std::make_unique<NiceMock<MockHTTPClient>>(std::move(ref));
    ON_CALL(*ptr, executeRequestImpl(_, _, _, _))
        .WillByDefault([=](const HTTPRequest &request, raw_ostream &os,
                           std::chrono::milliseconds timeout,
                           size_t maxLength) {
          requestUrls.emplace_back(request.URL);
          return HTTPResponse{};
        });
    return ptr;
  }

  void createContextWithHTTP(const MeteringContext::Options &options,
                             size_t maxProcessors = 8) {
    mockHttpClientRef = HTTPContext::init(
        [=](HTTPContextRef ref) { return createHTTPClient(std::move(ref)); });
    context =
        MeteringContext::create(options, mockHttpClientRef, maxProcessors);
    ASSERT_FALSE(context->setDefaultCallback(mockTelemetryCtx).isError());
  }

  void expectValuesAdded() {
    EXPECT_CALL(*mockTelemetryCtx.mockLogger, emitL0Event(_, _))
        .WillRepeatedly(
            [this](llvm::StringRef eventName,
                   const llvm::StringMap<Logs::AttributeValue> &attrs) {
              values.push_back(attrs);
            });
  }

  void expectStringValue(const llvm::StringMap<Logs::AttributeValue> &values,
                         StringRef key, StringRef str) const {
    ASSERT_TRUE(values.contains(key)) << "Missing key: " << key.str();
    const auto printed = llvm::print(values.at(key));
    EXPECT_EQ(printed, str) << "Wrong value for key: " << key.str();
  }

  void expectContainsKey(const llvm::StringMap<Logs::AttributeValue> &values,
                         StringRef key, bool contained = true) const {
    EXPECT_EQ(values.contains(key), contained) << "For key: " << key.str();
  }

  void expectFixedAttributes(
      const llvm::StringMap<Logs::AttributeValue> &values) const {
    expectStringValue(values, "event_type", MeteringContext::kEventType);
    expectStringValue(values, "cloud", "aws");
    expectStringValue(values, "region", "us-west-2");
    expectStringValue(values, "instance_class", "c5");
    expectStringValue(values, "instance_type", "c5.4xlarge");
  }

  // N.B. Initialize the underlying telemetry with empty settings.
  Settings empty =
      Settings(Config(), EntitlementStore::alwaysOpen(llvm::errs()));
  NiceMock<MockTelemetryContext> mockTelemetryCtx =
      NiceMock<MockTelemetryContext>(empty);
  std::unique_ptr<MeteringContext> context;
  std::vector<llvm::StringMap<Logs::AttributeValue>> values;

  HTTPContextRef mockHttpClientRef;
  std::vector<std::string> requestUrls;
};

TEST_F(MeteringContextLogTest, EmptyInstanceInfo) {
  EXPECT_CALL(*mockTelemetryCtx.mockLogger, emitL0Event(_, _))
      .Times(3) // start(), flush(), destructor flush()
      .WillRepeatedly(
          [this](llvm::StringRef eventName,
                 const llvm::StringMap<Logs::AttributeValue> &attributes) {
            values.push_back(attributes);
          });

  createContextWithHTTP({});
  // Smoke test that various cloud instance metadata endpoints are checked.
  EXPECT_THAT(requestUrls,
              testing::ElementsAre(
                  // AWS IMDS
                  "http://169.254.169.254/latest/api/token",
                  "http://169.254.169.254/latest/meta-data/placement/region",
                  "http://169.254.169.254/latest/meta-data/instance-type"));

  auto result = context->flush(); // Another in the callback registration.
  EXPECT_FALSE(result.isError());
  EXPECT_EQ(values.size(), 2u);
  expectContainsKey(values[0], "region", false);
  expectContainsKey(values[0], "instance_class", false);
  expectContainsKey(values[0], "instance_type", false);
}

TEST_F(MeteringContextLogTest, Flush) {
  expectValuesAdded();

  createContext({});
  std::this_thread::sleep_for(std::chrono::seconds(1));
  auto result = context->flush();

  EXPECT_FALSE(result.isError());
  EXPECT_EQ(values.size(), 2u);
  expectFixedAttributes(values[0]);
  expectStringValue(values[0], "cpu_seconds", "8");
  expectStringValue(values[1], "cpu_seconds", "8");
}

TEST_F(MeteringContextLogTest, FlushWithLimits) {
  expectValuesAdded();

  const auto maxProcessors = 4;
  createContext({}, maxProcessors);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  auto result = context->flush();

  EXPECT_FALSE(result.isError());
  EXPECT_EQ(values.size(), 2u);
  expectFixedAttributes(values[0]);
  expectStringValue(values[0], "cpu_seconds", std::to_string(maxProcessors));
  expectStringValue(values[1], "cpu_seconds", std::to_string(maxProcessors));
}
} // namespace M::Metering
