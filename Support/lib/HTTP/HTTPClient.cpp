//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HTTP/HTTPClient.h"

#include "Cache/BlobCache.h"
#include "Support/Base64.h"
#include "Support/Configuration.h"
#include "Support/Threading/Shared.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"
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

void HTTPContext::setShouldVerifyTLSPeer(bool verifyTLSPeer) {
  this->verifyTLSPeer = verifyTLSPeer;
}

void HTTPContext::setUserAgent(std::string userAgent) {
  this->userAgent = std::move(userAgent);
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

/// Clean out any auth settings we may have set already.
static void cleanupAuth(void *curl) {
  curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
  curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, nullptr);
  curl_easy_setopt(curl, CURLOPT_SSLCERT_BLOB, nullptr);
  curl_easy_setopt(curl, CURLOPT_SSLCERTTYPE, "PEM");
  curl_easy_setopt(curl, CURLOPT_SSLKEY_BLOB, nullptr);
  curl_easy_setopt(curl, CURLOPT_SSLKEYTYPE, "PEM");
}

void HTTPClient::noAuthNeeded() {
  cleanupAuth(curl);
  authSetup = true;
}

ErrorOrSuccess HTTPClient::setupAuth(std::optional<std::string> tok) {
  // Clean out anything we already had. This means just resetting to the
  // defaults so that we don't accidentally use an outdated token, for example.
  cleanupAuth(curl);

  // Short-circuit if we're using bearer token authorization. This way, libcurl
  // will add the headers *for* us.
  if (tok) {
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
    curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, tok->c_str());
    authSetup = true;
    return success();
  }

  auto clientCert = findModularFile("client.pem");
  if (!clientCert)
    return Error("could not find the client certificate");

  auto certBufOr = Buffer::getFile(*clientCert);
  if (certBufOr.isError())
    return certBufOr.takeError();

  // Set the client certificate on the context.
  curl_blob blob = {};
  blob.data = const_cast<void *>((const void *)(*certBufOr)->getBufferStart());
  blob.len = (*certBufOr)->getBufferSize();
  blob.flags = CURL_BLOB_COPY;
  curl_easy_setopt(curl, CURLOPT_SSLCERT_BLOB, &blob);
  curl_easy_setopt(curl, CURLOPT_SSLCERTTYPE, "PEM");

  auto clientKey = findModularFile("client_priv.pem");
  if (!clientKey)
    return Error("could not find the client private key");

  auto clientKeyBufOr = Buffer::getFile(*clientKey);
  if (clientKeyBufOr.isError())
    return clientKeyBufOr.takeError();

  // Now set the client key on the request - used to sign the request.
  blob.data =
      const_cast<void *>((const void *)(*clientKeyBufOr)->getBufferStart());
  blob.len = (*clientKeyBufOr)->getBufferSize();
  curl_easy_setopt(curl, CURLOPT_SSLKEY_BLOB, &blob);
  curl_easy_setopt(curl, CURLOPT_SSLKEYTYPE, "PEM");

  // All done!
  authSetup = true;
  return success();
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
  if (!authSetup)
    llvm::report_fatal_error("auth must be setup before executing a request");

  return executeRequestImpl(request, os, timeout, maxLength);
}

class ProgressWrapper {
public:
  ProgressWrapper(Progress *underlying) : progress(underlying) {}
  ~ProgressWrapper() {
    if (total > finished)
      progress->skippedBytes(total - finished);
  }
  void callback(curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal,
                curl_off_t ulnow) {
    // It's possible that the total will change over time (if e.g. no length is
    // provided and we have a chunked encoding), so we need to use the grow
    // call appropriately here.
    size_t cur_total =
        static_cast<size_t>(dltotal) + static_cast<size_t>(ultotal);
    if (cur_total > total) {
      progress->addBytes(cur_total - total);
      total = cur_total;
    }
    size_t new_finished =
        static_cast<size_t>(dlnow) + static_cast<size_t>(ulnow);
    if (new_finished > finished) {
      if (new_finished > total) {
        progress->addBytes(new_finished - total);
        total = new_finished;
      }
      progress->finishedBytes(new_finished - finished);
      finished = new_finished;
    }
  }

private:
  Progress *progress;
  size_t total = 0;
  size_t finished = 0;
};

static size_t progressCallback(void *clientp, curl_off_t dltotal,
                               curl_off_t dlnow, curl_off_t ultotal,
                               curl_off_t ulnow) {
  ProgressWrapper *wrapper = static_cast<ProgressWrapper *>(clientp);
  assert(wrapper);
  wrapper->callback(dltotal, dlnow, ultotal, ulnow);
  return 0;
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

  // If there is a progress function, set up the callback appropriately. We
  // pass the progress object itself as the callback data.
  ProgressWrapper progress(request.progress);
  if (request.progress) {
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0),
                     "set no progress");
    CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_XFERINFODATA,
                                      static_cast<void *>(&progress)),
                     "set progress handler");
    CHECK_CURL_ERROR(
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback),
        "set progress callback");
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
  // Verify SSL certificate against peers
  CHECK_CURL_ERROR(curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER,
                                    context->verifyTLSPeer ? 1 : 0),
                   "set ssl verify peer");
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
