//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HTTP/HTTPClient.h"

#include "Support/Threading/Shared.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "curl/curl.h"

#include <chrono>

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
  // Flush/free all caches and close persistant connections
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

struct RequestStreamReturn {
  llvm::raw_ostream *os;
  size_t limit = 0;
  size_t written = 0;
};

static size_t streamWriter(char *contents, size_t size, size_t members,
                           void *stream) {
  RequestStreamReturn *ret = static_cast<RequestStreamReturn *>(stream);
  size_t len = size *= members; // size is always 1 in CURL for legacy reasons.
  if (ret->limit > 0 && ret->written + len > ret->limit)
    len = ret->limit - ret->written;
  ret->os->write(contents, len);
  return len;
}

/// Adapt an llvm::unique_function to a libcurl read callback.
static size_t readCallback(char *buffer, size_t size, size_t nitems,
                           void *userdata) {
  auto *fn = (HTTPRequest::ReadCallback *)userdata;
  auto sizeOr = (*fn)(buffer, size * nitems);
  if (sizeOr.isError())
    return CURL_READFUNC_ABORT;

  return *sizeOr;
}

HTTPResponse HTTPClient::executeRequest(const HTTPRequest &request,
                                        raw_ostream &os,
                                        std::chrono::milliseconds timeout,
                                        size_t maxLength) {
  return executeRequestImpl(request, os, timeout, maxLength);
}

HTTPResponse HTTPClient::executeRequestImpl(const HTTPRequest &request,
                                            raw_ostream &os,
                                            std::chrono::milliseconds timeout,
                                            size_t maxLength) {
  RequestStreamReturn ret;
  ret.os = &os;
  ret.limit = maxLength;
  ret.written = 0;

  // Handle the headers. This will format them as "key: value".
  curl_slist *list = nullptr;
  auto freeList = llvm::make_scope_exit([&] { curl_slist_free_all(list); });
  for (const auto &h : request.headers) {
    list = curl_slist_append(
        list, llvm::formatv("{0}: {1}", h.first(), h.second).str().c_str());
  }

  // TODO arekay - figure out what are invalid combinations. The port is encoded
  // in the url, so it is an overkill to check if the url contains a port as
  // well as uds being set.
  if (!request.udsName.empty()) {
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_UNIX_SOCKET_PATH,
                                      request.udsName.c_str()),
                     "set uds");
  }

  // Set total timeout.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout.count()),
                   "set timeout");
  // Set connection timeout to 20 seconds.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L),
                   "set connection timeout");
  // Set the headers.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list),
                   "set http header list");

  // Set the token, if provided.
  if (request.accessToken) {
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
    curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER,
                     request.accessToken->c_str());
  } else {
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
    curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, nullptr);
  }

  // Setup the key; note the CURL_BLOB_COPY means that it will not touch this
  // data, and the const cast is harmless.
  if (context->clientKey.size() > 0) {
    curl_blob blob = {};
    blob.data = const_cast<void *>((const void *)(context->clientKey.c_str()));
    blob.len = context->clientKey.size();
    blob.flags = CURL_BLOB_COPY;
    curl_easy_setopt(curl, CURLOPT_SSLKEY_BLOB, &blob);
    curl_easy_setopt(curl, CURLOPT_SSLKEYTYPE, "PEM");
  } else {
    curl_easy_setopt(curl, CURLOPT_SSLKEY_BLOB, nullptr);
  }

  // Setup the cert; see above re: const_cast.
  if (context->clientCert.size() > 0) {
    curl_blob blob = {};
    blob.data = const_cast<void *>((const void *)(context->clientCert.c_str()));
    blob.len = context->clientCert.size();
    blob.flags = CURL_BLOB_COPY;
    curl_easy_setopt(curl, CURLOPT_SSLCERT_BLOB, &blob);
    curl_easy_setopt(curl, CURLOPT_SSLCERTTYPE, "PEM");
  } else {
    curl_easy_setopt(curl, CURLOPT_SSLCERT_BLOB, nullptr);
  }

  // Set verbose mode if running in DEBUG mode
  CHECK_CURL_ERROR(
      curl_easy_setopt(curl, CURLOPT_VERBOSE, llvm::DebugFlag ? 1 : 0),
      "set verbose mode");

  // Set the method.
  switch (request.method) {
  case HTTPRequest::GET:
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_HTTPGET, 1),
                     "set header to get");
    break;
  case HTTPRequest::PUT:
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_PUT, 1),
                     "set header to put");
    break;
  case HTTPRequest::POST:
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_POST, 1),
                     "set header to post");
    break;
  case HTTPRequest::HEAD:
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_HTTPGET, 1),
                     "set header to head.httpget");
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_NOBODY, 1),
                     "set header to head nobody");
    break;
  }

  // We can set the read data as a callback.
  if (request.body) {
    CHECK_CURL_ERROR(
        curl_easy_setopt(curl, CURLOPT_READFUNCTION, &readCallback),
        "set body callback");
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_READDATA, &request.body),
                     "set body");

    // If the user provided the full size of the data here, provide it to curl.
    if (request.bodyLen)
      CHECK_CURL_ERROR(
          curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, *request.bodyLen),
          "set body len");
  }

  // Set URL we will perform the HTTP
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_URL, request.URL.c_str()),
                   "set url");
  // Follow any HTTP 301 or 302  redirects implicity.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, true),
                   "set follow location");
  // Set our write callback function.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &streamWriter),
                   "set write function");
  // Set our user data object for our callback.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ret),
                   "set write data");
  // Allow transport compression. Empty string means all supported.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, ""),
                   "set accept encoding");
  // Verify SSL certificate against peers.
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1),
                   "set ssl verify peer");
  // Set the CA info if appropriate.
  if (!context->caInfo.empty())
    CHECK_CURL_ERROR(
        curl_easy_setopt(curl, CURLOPT_CAINFO, context->caInfo.c_str()),
        "set ssl cainfo");
  // Let the server know who we are.
  CHECK_CURL_ERROR(
      curl_easy_setopt(curl, CURLOPT_USERAGENT, context->userAgent.c_str()),
      "set user agent");

  if (request.range) {
    auto [start, end] = *request.range;
    if (end) {
      auto str = llvm::formatv("{0}-{1}", start, *end);
      curl_easy_setopt(curl, CURLOPT_RANGE, str.str().c_str());
    } else {
      auto str = llvm::formatv("{0}-", start);
      curl_easy_setopt(curl, CURLOPT_RANGE, str.str().c_str());
    }
  } else {
    curl_easy_setopt(curl, CURLOPT_RANGE, nullptr);
  }

  // Execute our reqeust.
  CURLcode res = curl_easy_perform(curl);

  HTTPResponse response;

  if (res != CURLE_OK) {
    if (res == CURLE_OPERATION_TIMEDOUT)
      response.kind = HTTPResponse::Kind::TimeoutError;
    else
      response.kind = HTTPResponse::Kind::TransportError;

    response.transportErrorMessage =
        llvm::formatv("failed to reach URL {0} with cURL error {1}",
                      request.URL, curl_easy_strerror(res));
  } else {
    // Check our response code.
    long responseCode;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);

    response.responseCode = responseCode;

    if (responseCode >= 400 && responseCode <= 599)
      response.kind = HTTPResponse::Kind::HTTPResponseError;
    else
      response.kind = HTTPResponse::Kind::Success;
  }

  return response;
}
