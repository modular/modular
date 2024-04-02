//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H
#define SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H

#include "Support/HTTP/HTTPClient.h"

namespace M::Metering {

/// Identifies AWS region and instance type by querying the IMDSv2 HTTP API.
class InstanceIdentifier {
public:
  InstanceIdentifier(HTTPContextRef ctx) : client(ctx->client()) {}

  ErrorOrSuccess fetch();

  StringRef getRegion() const { return region; }
  StringRef getInstanceType() const { return instanceType; }

private:
  ErrorOrSuccess fetchInfo(StringRef token, bool isIPv4);
  ErrorOrSuccess fetchV1();
  ErrorOrSuccess fetchV2();

  std::unique_ptr<HTTPClient> client;
  std::string region;
  std::string instanceType;
};

} // namespace M::Metering

#endif // SUPPORT_METERING_AWS_INSTANCEIDENTIFIER_H
