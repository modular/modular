//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Metering/AWS/InstanceIdentifier.h"

#include "Support/Buffer.h"
#include "Support/LLVMForwardDecls.h"

#define DEBUG_TYPE "modular-metering"

namespace M::Metering {
namespace {

// The IMDS static endpoints are documented at:
// https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
constexpr StringRef kIPv4BaseUrl = "http://169.254.169.254";
constexpr StringRef kIPv6BaseUrl = "http://[fd00:ec2::254]";

constexpr StringRef kTokenSuffix = "/latest/api/token";
constexpr StringRef kRegionSuffix = "/latest/meta-data/placement/region";
constexpr StringRef kInstanceTypeSuffix = "/latest/meta-data/instance-type";

constexpr StringRef kTokenTTLHeader = "X-aws-ec2-metadata-token-ttl-seconds";
constexpr StringRef kTokenHeader = "X-aws-ec2-metadata-token";

ErrorOr<size_t> emptyRead(char *buffer, size_t bytes) { return 0; }

HTTPRequest defaultRequest(StringRef url) {
  HTTPRequest req{url.str()};
  req.headers["Expect"] = "";
  req.headers["Transfer-Encoding"] = "";
  return req;
}

HTTPRequest tokenRequest(StringRef baseUrl) {
  HTTPRequest req = defaultRequest((baseUrl + kTokenSuffix).str());
  req.method = HTTPRequest::Method::PUT;
  req.headers[kTokenTTLHeader] = "21600";
  req.body = emptyRead;
  return req;
}

HTTPRequest regionRequest(StringRef baseUrl, StringRef token = "") {
  HTTPRequest req = defaultRequest((baseUrl + kRegionSuffix).str());
  if (!token.empty())
    req.headers[kTokenHeader] = token;
  req.body = emptyRead;
  return req;
}

HTTPRequest instanceTypeRequest(StringRef baseUrl, StringRef token = "") {
  HTTPRequest req = defaultRequest((baseUrl + kInstanceTypeSuffix).str());
  if (!token.empty())
    req.headers[kTokenHeader] = token;
  req.body = emptyRead;
  return req;
}

} // namespace

ErrorOr<std::string> InstanceIdentifier::send(const HTTPRequest &req) {
  auto writeBuf = WriteableBuffer::get();
  HTTPResponse response = client->executeRequest(
      req, *writeBuf,
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
  if (response.isError())
    return response.asError().takeError();
  return writeBuf->getBuffer().data();
}

ErrorOrSuccess InstanceIdentifier::fetchInfo(StringRef token, bool isIPv4) {
  const auto &baseURL = isIPv4 ? kIPv4BaseUrl : kIPv6BaseUrl;
  auto regionOr = send(regionRequest(baseURL, token));
  if (regionOr.isError())
    return regionOr.takeError();

  auto instanceTypeOr = send(instanceTypeRequest(baseURL, token));
  if (instanceTypeOr.isError())
    return instanceTypeOr.takeError();

  // All or nothing.
  region = *regionOr;
  instanceType = *instanceTypeOr;
  return success();
}

ErrorOrSuccess InstanceIdentifier::fetchV1() {
  if (auto resultV4 = fetchInfo("", true); resultV4.isError()) {
    if (auto resultV6 = fetchInfo("", false); resultV6.isError()) {
      return Error(
          llvm::Twine("Could not reach the IMDSv1 API with errors: {") +
          resultV4.getError() + " | " + resultV6.getError() + "}");
    }
  }
  return success();
}

ErrorOrSuccess InstanceIdentifier::fetchV2() {
  // Try IPv4 first and fallback to IPv6.
  bool isIPv4 = true;
  std::string token;
  if (auto resultV4 = send(tokenRequest(kIPv4BaseUrl)); resultV4.isError()) {
    isIPv4 = false;
    if (auto resultV6 = send(tokenRequest(kIPv6BaseUrl)); resultV6.isError()) {
      return Error(
          llvm::Twine("Could not reach the IMDSv2 API with errors: {") +
          resultV4.getError() + " | " + resultV6.getError() + "}");
    } else
      token = *resultV6;
  } else
    token = *resultV4;
  LLVM_DEBUG(llvm::dbgs() << "IMDSv2 token: " << token << "\n");
  return fetchInfo(token, isIPv4);
}

ErrorOrSuccess InstanceIdentifier::fetch() {
  if (auto resultV2 = fetchV2(); resultV2.isError()) {
    if (auto resultV1 = fetchV1(); resultV1.isError()) {
      return Error(llvm::Twine("Could not reach any IMDS API with errors: {") +
                   resultV2.getError() + " | " + resultV1.getError() + "}");
    }
    LLVM_DEBUG(llvm::dbgs() << "IMDSv1 response: (" << region << ", "
                            << instanceType << ")\n");
    return success();
  }
  LLVM_DEBUG(llvm::dbgs() << "IMDSv2 response: (" << region << ", "
                          << instanceType << ")\n");
  return success();
}

} // namespace M::Metering
