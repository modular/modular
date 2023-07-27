//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HTTPClient.h"

#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;

HTTPContextRef HTTPContext::init() {
  static std::atomic_flag initialized = ATOMIC_FLAG_INIT;
  if (initialized.test_and_set())
    assert(false && "A HTTPContext can only be initialize once in a process.");
  return HTTPContextRef::create();
}

HTTPContext::HTTPContext() {
  // Warm up cURL's SSL backend, resolver cache, logging, etc
  curl_global_init(CURL_GLOBAL_ALL);
}

HTTPContext::~HTTPContext() {
  // Flush/free all caches and close persistant connections
  curl_global_cleanup();
}

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

ErrorOrSuccess HTTPClient::executeRequest(const HTTPRequest &request,
                                          raw_ostream &os,
                                          std::chrono::milliseconds timeout,
                                          size_t maxLength) {

  RequestStreamReturn ret;
  ret.os = &os;
  ret.limit = maxLength;
  ret.written = 0;

  // Set HTTP Request timeout.
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout.count());
  // For now we will only do HTTP GET requests.
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1);
  // Set URL we will perform the HTTP
  curl_easy_setopt(curl, CURLOPT_URL, request.URL.c_str());
  // Follow any HTTP 301 or 302  redirects implicity.
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, true);
  // Set our write callback function.
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &streamWriter);
  // Set our user data object for our callback.
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ret);
  // Verify SSL certificate against peers
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER,
                   request.verify_tls_peer ? 1 : 0);

  // Execute our reqeust.
  CURLcode res = curl_easy_perform(curl);

  if (res != CURLE_OK) {
    return Error(llvm::formatv("failed to reach URL {0} with cURL error {1}",
                               request.URL, curl_easy_strerror(res)));
  }
  return success();
}
