//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HTTP/HTTPClient.h"

#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/Threading/Shared.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "curl/curl.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <curl/easy.h>
#include <memory>
#include <utility>

#define CHECK_CURL_ERROR(X, MSG)                                               \
  {                                                                            \
    do {                                                                       \
      auto errorCode = (X);                                                    \
      if (errorCode) {                                                         \
        HTTPResponse response;                                                 \
        response.kind = M::HTTPResponse::TransportError;                       \
        llvm::errs() << "Error in curl call " << MSG << " : \'"                \
                     << curl_easy_strerror(errorCode) << "\'\n";               \
        response.transportErrorMessage = curl_easy_strerror(errorCode);        \
        return response;                                                       \
      } /*if*/                                                                 \
    } while (false);                                                           \
  }

using namespace M;

//===----------------------------------------------------------------------===//
// HTTPContext
//===----------------------------------------------------------------------===//

HTTPContextRef HTTPContext::init(ClientConstructor cc) {
  // Ideally this would only be called once. Failing that, multiple calls are
  // safe but only the first call will apply. See:
  // https://curl.se/libcurl/c/curl_global_init.html.
  auto httpCtx = HTTPContextRef::create();
  httpCtx->cc = std::move(cc);
  return httpCtx;
}

HTTPContext::HTTPContext() : cc(nullptr), userAgent("modular-installer/0.1") {
  // Warm up cURL's SSL backend, resolver cache, logging, etc
  curl_global_init(CURL_GLOBAL_ALL);
}

std::unique_ptr<HTTPClient> HTTPContext::client() {
  if (cc)
    return cc(HTTPContextRef::copy(this));
  return std::make_unique<HTTPClient>(HTTPContextRef::copy(this));
}

void HTTPContext::setUserAgent(std::string userAgent) {
  this->userAgent = std::move(userAgent);
}

void HTTPContext::setupAuth(std::string clientKey, std::string clientCert) {
  this->clientKey = clientKey;
  this->clientCert = clientCert;
}

void HTTPContext::setCAInfo(std::string caInfo) {
  this->caInfo = std::move(caInfo);
}

HTTPContext::~HTTPContext() {
  // Flush/free all caches and close persistent connections
  curl_global_cleanup();
}

ErrorOrSuccess HTTPResponse::asError(StringRef extraContext) {
  switch (kind) {
  case Success:
    return success();
  case TransportError:
  case TimeoutError:
    assert(transportErrorMessage && "current error is not set");
    if (extraContext.empty())
      return Error("http error: " + *transportErrorMessage);
    return Error("http error: " + *transportErrorMessage + " - " +
                 extraContext);
  case HTTPResponseError:
    assert(responseCode && "responseCode is not set");
    if (extraContext.empty())
      return Error(
          llvm::formatv("http error: response code {0}", responseCode).str());
    return Error(llvm::formatv("http error: response code {0} - {1}",
                               responseCode, extraContext)
                     .str());
  }
  llvm_unreachable("Invalid response kind.");
}

//===----------------------------------------------------------------------===//
// HTTPClient
//===----------------------------------------------------------------------===//

HTTPClient::HTTPClient(HTTPContextRef ctx) : context(std::move(ctx)) {
  curl = curl_easy_init();
  assert(curl && "libcurl could not be initialized");
}

HTTPClient::~HTTPClient() {
  if (curl) {
    curl_easy_cleanup(curl);
    curl = nullptr;
  }
}
