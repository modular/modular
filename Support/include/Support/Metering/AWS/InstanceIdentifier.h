//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H
#define SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H

#include "Support/HTTP/HTTPClient.h"

namespace M::Metering {

/// Identifies AWS region and instance type by querying the IMDSv{1, 2} API.
class InstanceIdentifier {
public:
  using ClockType = std::chrono::steady_clock;
  using DurationType = std::chrono::duration<ClockType::rep, ClockType::period>;

  InstanceIdentifier(HTTPContextRef ctx) : client(ctx->client()) {}

  ErrorOrSuccess fetch();

  StringRef getRegion() const { return region; }
  StringRef getInstanceType() const { return instanceType; }

private:
  ErrorOr<std::string> send(const HTTPRequest &req);

  ErrorOrSuccess fetchInfo(StringRef token, bool isIPv4);
  ErrorOrSuccess fetchV1();
  ErrorOrSuccess fetchV2();

  const DurationType timeout = std::chrono::seconds(1);

  std::unique_ptr<HTTPClient> client;
  std::string region;
  std::string instanceType;
};

} // namespace M::Metering

#endif // SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H
