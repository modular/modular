//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HTTPClient.h"

#include "Cache/BlobCache.h"
#include "Support/Base64.h"
#include "Support/Threading/Shared.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/FormatVariadic.h"

#include "curl/curl.h"

#include <chrono>

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

ErrorOrSuccess HTTPResponse::asError(StringRef extraContext) {
  switch (kind) {
  case Success:
    return success();
  case TransportError:
    assert(transportErrorMessage && "current error is not set");
    return Error("http error: " + *transportErrorMessage + " - " +
                 extraContext);
  case HTTPResponseError:
    assert(responseCode && "responseCode is not set");
    return Error(llvm::formatv("http error: response code {0} - {1}",
                               responseCode, extraContext)
                     .str());
  }
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

HTTPResponse HTTPClient::executeRequest(const HTTPRequest &request,
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
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, request.verifyTLSPeer ? 1 : 0);
  // Let the server know who we are.
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "modular-installer/0.1");

  // Execute our reqeust.
  CURLcode res = curl_easy_perform(curl);

  HTTPResponse response;

  if (res != CURLE_OK) {
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

//===----------------------------------------------------------------------===//
// HTTPCASBackend Implementation
//===----------------------------------------------------------------------===//

ErrorOrSuccess HTTPCASBackend::insertImpl(StringRef keyHash,
                                          Cache::BufferRef obj) {
  return Error::getStaticString("HTTP backend does not support insert");
}

ErrorOr<bool> HTTPCASBackend::containsImpl(StringRef keyHash) const {
  auto findOr = findImpl(keyHash, std::nullopt);
  if (findOr.isError())
    return findOr.takeError();

  return findOr->has_value();
}

ErrorOr<std::optional<Cache::BufferRef>>
HTTPCASBackend::findImpl(StringRef keyHash,
                         std::optional<Cache::WriteableBufferRef> buf) const {
  // First, check to see if we've already got this in our local cache.
  auto localBuf = const_cast<HTTPCASBackend *>(this)->findBuffer(keyHash);
  if (localBuf) {
    // If we weren't passed a buffer, return the ref we have.
    if (!buf)
      return std::move(*localBuf);

    // If we were, then write the contents of our local buffer into the thing we
    // were given.
    (*buf)->write((*localBuf)->getBufferStart(), (*localBuf)->getBufferSize());
    return std::move(*buf);
  }

  // Base64 encode the key hash so it's URL-safe.
  std::string keyHashB64 = encodeURLSafeBase64(keyHash);

  // We didn't have it locally, so create a request to go get it.
  HTTPClient client(ctx.copy());
  HTTPRequest req{
      /*URL=*/url + "/" + keyHashB64,
      /*verifyTLSPeer=*/true,
  };

  // 60s timeout for requests.
  using namespace std::chrono_literals;
  constexpr std::chrono::milliseconds timeout = 60s;
  // Maximum of 512M per request.
  constexpr size_t maxBytes = 1024 * 1024 * 512;

  // Either create a new WriteableBuffer or use the one that was passed in.
  auto writeBuf =
      buf.has_value()
          ? std::move(*buf)
          : Cache::WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                        /*capacity=*/maxBytes);

  // Execute the request.
  HTTPResponse response =
      client.executeRequest(req, *writeBuf, timeout, maxBytes);

  // TODO: Will the result bytes be encoded or can we expect them to be raw
  //       binary?

  // Everything is fine, return the buffer.
  if (response.isSuccess()) {
    // Cache it, so we can avoid multiple requests at this level.
    const_cast<HTTPCASBackend *>(this)->cacheBuffer(keyHash, writeBuf.copy());
    return std::move(writeBuf);
  }

  // Content was not found - this is not an error, just return nullopt.
  if (response.isError() && response.responseCode &&
      *response.responseCode == HTTPResponseCode::NotFound)
    return std::nullopt;

  // Return the error we hit.
  std::string errorContextStr =
      llvm::formatv("Looking for {0}", keyHashB64).str();
  return response.asError(errorContextStr).takeError();
}

ErrorOrSuccess HTTPCASBackend::clearImpl() {
  return Error::getStaticString("HTTP backend does not support clear");
}

void HTTPCASBackend::cacheBuffer(StringRef keyHash, Cache::BufferRef buf) {
  localCache.modify([&](llvm::StringMap<Cache::BufferRef> &map) {
    map[keyHash] = std::move(buf);
  });
}

std::optional<Cache::BufferRef> HTTPCASBackend::findBuffer(StringRef keyHash) {
  return localCache.read([&](llvm::StringMap<Cache::BufferRef> &map)
                             -> std::optional<Cache::BufferRef> {
    auto found = map.find(keyHash);
    if (found == map.end())
      return std::nullopt;
    return found->second.copy();
  });
}

HTTPCASBackendRef M::getHTTPCASBackend(HTTPContextRef ctx, std::string url,
                                       Runtime &runtime) {
  return HTTPCASBackendRef::create(std::move(ctx), std::move(url), runtime);
}
