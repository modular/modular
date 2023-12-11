//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODELSERVING_AWS_INSTANCEIDENTIFIER_H
#define MODELSERVING_AWS_INSTANCEIDENTIFIER_H

#include "Support/HTTP/HTTPClient.h"

namespace M::ModelServing::Billing {

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

} // namespace M::ModelServing::Billing

#endif // MODELSERVING_AWS_INSTANCEIDENTIFIER_H
