//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULAR_COMMON_HTTPCLIENT_H
#define MODULAR_COMMON_HTTPCLIENT_H

#include "Cache/BlobCache.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/Threading/Shared.h"
#include <filesystem>
#include <string>

namespace M {

class HTTPContext;
using HTTPContextRef = RCRef<HTTPContext>;
/// Provides a presistant HTTP context to create HTTPClients.
/// Initializes and cleans up the global CURL initalization.
/// Ideally should scope to your applications main method.
class HTTPContext : public ReferenceCounted<HTTPContext> {
public:
  ~HTTPContext();

  /// Initialize an HTTP context.
  static HTTPContextRef init();

protected:
  // Allow access to protected constructor.
  friend class RCRef<HTTPContext>;

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
  ErrorOrSuccess asError(StringRef extraContext = "");
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
} // namespace M

#endif // MODULAR_COMMON_HTTPCLIENT_H
