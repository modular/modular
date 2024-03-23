//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BILLING_AWS_INSTANCEIDENTIFIER_H
#define SUPPORT_BILLING_AWS_INSTANCEIDENTIFIER_H

#include "Support/HTTP/HTTPClient.h"

namespace M::Billing {

/// Identifies AWS region and instance type by querying the IMDSv2 HTTP API.
class InstanceIdentifier {
public:
  InstanceIdentifier(HTTPContextRef c) : ctx(std::move(c)) {}

  ErrorOrSuccess fetch();

  StringRef getRegion() const { return region; }
  StringRef getInstanceType() const { return instanceType; }

private:
  ErrorOrSuccess fetchInfo(HTTPClient &client, StringRef token, bool isIPv4);
  ErrorOrSuccess fetchV1(HTTPClient &client);
  ErrorOrSuccess fetchV2(HTTPClient &client);

  HTTPContextRef ctx;
  std::string region;
  std::string instanceType;
};

} // namespace M::Billing

#endif // SUPPORT_BILLING_AWS_INSTANCEIDENTIFIER_H
