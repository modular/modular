//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULAR_COMMON_HTTPCLIENT_H
#define MODULAR_COMMON_HTTPCLIENT_H

#include "Cache/BlobCache.h"
#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"
#include "Support/Threading/Shared.h"
#include <filesystem>
#include <string>

namespace M {

class HTTPContext;
using HTTPContextRef = LLCL::RCRef<HTTPContext>;
/// Provides a presistant HTTP context to create HTTPClients.
/// Initializes and cleans up the global CURL initalization.
/// Ideally should scope to your applications main method.
class HTTPContext : public LLCL::ReferenceCounted<HTTPContext> {
public:
  ~HTTPContext();

  /// Initialize an HTTP context.
  static HTTPContextRef init();

protected:
  // Allow access to protected constructor.
  friend class LLCL::RCRef<HTTPContext>;

  HTTPContext();
};

/// Represents an HTTP Request
///
/// TODO: Add support for passing HTTP headers, setting methods, etc.
struct HTTPRequest {
  // Request URL.
  std::string URL;
  // Used to disable HTTPS vertification. Typically used to test with self
  // signed certicates.
  bool verifyTLSPeer = true;
};

/// Typical HTTP response code errors.
enum HTTPResponseCode : long {
  // Client Errors
  BadRequest = 400,
  Unauthorized = 401,
  PaymentRequired = 402,
  Forbidden = 403,
  NotFound = 404,
  MethodNotAllowed = 405,
  NotAcceptable = 406,
  ProxyAuthenticationRequired = 407,
  RequestTimeout = 408,
  Conflict = 409,
  Gone = 410,
  LengthRequired = 411,
  PreconditionFailed = 412,
  PayloadTooLarge = 413,
  URITooLong = 414,
  UnsupportedMediaType = 415,
  RangeNotSatisfiable = 416,
  ExpectationFailed = 417,
  ImATeapot = 418,
  MisdirectedRequest = 421,
  UpgradeRequired = 426,
  PreconditionRequired = 428,
  TooManyRequests = 429,
  UnavailableForLegalReasons = 451,

  // Server Errors (temporary errors)
  InternalServerError = 500,
  NotImplemented = 501,
  BadGateway = 502,
  ServiceUnavailable = 503,
  GatewayTimeout = 504,
  HTTPVersionNotSupported = 505,
};

/// HTTPResponse
struct HTTPResponse {
  enum Kind {
    Success,
    TransportError,    // Transport or CURL Error.
    HTTPResponseError, // Response code was not "200 Success" but an error of
                       // 4XX-5XX.
  } kind;

  // Can be compared against HTTPResponseCode for common HTTP Response Codes.
  std::optional<long> responseCode;

  // If we have CURL Error
  std::optional<std::string> transportErrorMessage;

  bool isSuccess() { return kind == Kind::Success; }
  bool isError() { return kind != Kind::Success; }
  ErrorOrSuccess asError();
};

/// HTTPClient that wraps libcurl.
///
/// Thead safety: HTTPClient is thread safe but doesn't provide synchronization.
class HTTPClient {
public:
  HTTPClient(HTTPContextRef ctx);
  ~HTTPClient();

  /// Blocking call that executes the HTTPRequest and writes the response to the
  /// provided ostream. Returns a HTTPResponse.
  ///
  /// Request `timeout` and `maxLength` can specified to limit requests.
  /// A `timeout` and `maxLength` of zero will not limit the request.
  HTTPResponse executeRequest(
      const HTTPRequest &request, raw_ostream &os,
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
      size_t maxLength = 0);

private:
  HTTPContextRef context;
  void *curl = nullptr;
};

//===----------------------------------------------------------------------===//
// HTTPCASBackend
//===----------------------------------------------------------------------===//

/// This class provides a wrapper around libcurl that conforms to the
/// BlobCacheBackend interface. This enables the CAS to interact with an HTTP
/// upstream.
// TODO: Once this is fully supported, graduate it to the Cache module.
class HTTPCASBackend : public Cache::BlobCacheBackend {
public:
  HTTPCASBackend(HTTPContextRef c, std::string url, LLCL::Runtime &runtime)
      : Cache::BlobCacheBackend(runtime), ctx(std::move(c)),
        url(std::move(url)) {}

  /// Insert a binary blob with a PUT request.
  // TODO: Currently unsupported
  ErrorOrSuccess insertImpl(StringRef keyHash, Cache::BufferRef obj) override;

  /// Check if we have the object we're looking for. Since really the only way
  /// to check if we have the thing is to fetch it, we just get it and trust our
  /// in-memory request cache.
  ErrorOr<bool> containsImpl(StringRef keyHash) const override;

  /// Find the object at `keyHash`. This essentially produces a GET request to
  /// <url>/<urlsafe-b64-key-hash>, which is expected to return the bytes of the
  /// object directly.
  ErrorOr<std::optional<Cache::BufferRef>>
  findImpl(StringRef keyHash,
           std::optional<Cache::WriteableBufferRef> buf) const override;

  /// We do not support clear for this backend impl.
  ErrorOrSuccess clearImpl() override;

private:
  /// The HTTP context ref for any clients we spin up for a find operation.
  HTTPContextRef ctx;
  /// The base URL for requests.
  std::string url;

  /// Store a buffer in our local in-memory cache. We can't always rely on the
  /// backends ahead of us for this, because our `containsImpl` just runs
  /// `findImpl` manually. If we have a find first though, backends before us
  /// (notably the in-memory backend) will perform this same function.
  void cacheBuffer(StringRef keyHash, Cache::BufferRef buf);

  /// Check if we have a buffer in our local in-memory cache. If we do, return
  /// it.
  std::optional<Cache::BufferRef> findBuffer(StringRef keyHash);

  /// Local cache we can use to store buffers we've fetched from upstream. This
  /// is important because of the way we had to implement `containsImpl` - HTTP
  /// doesn't really provide a way to check if a resource exists without
  /// just...getting the resource.
  Shared<llvm::StringMap<Cache::BufferRef>> localCache;
};

using HTTPCASBackendRef = LLCL::RCRef<HTTPCASBackend>;

/// Forward-declaration for the LLCL::Runtime.
namespace LLCL {
class Runtime;
}

/// Get the HTTP CAS backend. The backend will use `url` as the base, and append
/// the url-safe base-64 encoded key hash as the resource path.
HTTPCASBackendRef getHTTPCASBackend(HTTPContextRef ctx, std::string url,
                                    LLCL::Runtime &runtime);

} // namespace M

#endif // MODULAR_COMMON_HTTPCLIENT_H
