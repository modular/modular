//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HTTP_HTTPCLIENT_H
#define SUPPORT_HTTP_HTTPCLIENT_H

#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "Support/Threading/Shared.h"
#include "Support/UI/DataProgressBar.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include <filesystem>
#include <string>

namespace M {
/// Convenience declarations.
class HTTPContext;
class HTTPClient;
using HTTPContextRef = RCRef<HTTPContext>;

/// Provides a ref-counted HTTP context to create HTTPClients. Initializes and
/// cleans up the global CURL initialization. Ideally should scope to your
/// application's main method (similar to the AsyncRT::Runtime).
class HTTPContext : public ReferenceCounted<HTTPContext> {
public:
  ~HTTPContext();

  /// HTTPContext can be initialized with a default constructor for the client.
  /// This allows dependency injection in some tests.
  using ClientConstructor =
      std::function<std::unique_ptr<HTTPClient>(HTTPContextRef)>;

  /// Initialize an HTTP context.
  static HTTPContextRef init(ClientConstructor cc = nullptr);

  /// Set up the auth for this HTTP client. This can be set as often as desired.
  void setupAuth(std::string clientKey, std::string clientCert);

  std::unique_ptr<HTTPClient> client();
  void setUserAgent(std::string userAgent);
  void setCAInfo(std::string caInfo);

protected:
  /// Allow access to protected constructor.
  friend class RCRef<HTTPContext>;

  HTTPContext();

private:
  friend class HTTPClient;

  /// Constructor for new clients, may be nullptr.
  ClientConstructor cc;

  /// User agent to use for all requests.
  std::string userAgent;

  /// Client key for all requests.
  std::string clientKey;

  /// Client cert for all requests.
  std::string clientCert;

  /// caInfo contains certificate information (if non-empty).
  std::string caInfo;
};

/// Represents an HTTP Request.
struct HTTPRequest {
  /// Default constructor.
  HTTPRequest() = default;

  /// Construct an HTTPRequest from a Twine - makes it easier to construct URLs
  /// from multiple components.
  HTTPRequest(const Twine &t) : URL(t.str()) {}

  /// Request URL.
  std::string URL;

  /// An optional bearer token.
  std::optional<std::string> accessToken;

  /// UDS path if applicable - empty means disabled
  std::string udsName;

  /// Headers to set on the request.
  llvm::StringMap<std::string> headers = {};

  /// Method to use.
  enum Method {
    POST,
    PUT,
    GET,
    HEAD,
  };
  Method method = Method::GET;

  /// Range to request. If set, the server should return only the bytes in the
  /// range. This is useful for resuming downloads.
  /// The first element is the start of the range, and the second element is the
  /// end of the range. If the second element is `std::nullopt`, then the range
  /// is open-ended.
  std::optional<std::pair<size_t, std::optional<size_t>>> range = std::nullopt;

  /// curl generally recommends sending body data (when it's large) with a
  /// callback. This allows the user to specify any state that may need to be
  /// held. Write as much data as possible into `buffer`, but not more than
  /// `bytes` bytes. The callback will be called until it returns 0 - returning
  /// 0 signals EOF and the callback won't be called again. The callback may
  /// also return an error, in which case we will abort the transfer.
  using ReadCallback =
      llvm::unique_function<ErrorOr<size_t>(char *buffer, size_t bytes)>;
  ReadCallback body = nullptr;
  /// If you know exactly how many bytes you want to send up front, set this
  /// field. This allows libcurl to avoid some length checking.
  std::optional<size_t> bodyLen = std::nullopt;

  /// Potential progress callbacks.
  std::optional<DataProgressBar *> progress = std::nullopt;
};

/// This struct provides a libcurl-compatible read adaptor for a container that
/// has random access iterators. The container must be a container of single
/// bytes (uint8_t, char, etc).
template <typename T>
struct ContainerReadCallbackAdaptor {
  explicit ContainerReadCallbackAdaptor(T &container)
      : iter(container.begin()), end(container.end()) {}

  /// Store the beginning and end of the container. We don't need the actual
  /// container itself.
  typename T::iterator iter;
  typename T::iterator end;

  /// This is the actual callback - if there's nothing to read, it returns 0. If
  /// there's something to read, then it will read as much as possible into
  /// `buffer`, up to `bytes` bytes.
  ErrorOr<size_t> operator()(char *buffer, size_t bytes) {
    // Nothing left to copy in, so just return zero.
    if (iter == end)
      return 0;
    // Error case, iter > end. Unclear how it happened, but this *is* an error.
    if (iter > end)
      return Error("iter was incremented past the end");

    // The end pointer is the minimum of `bodyIter + bytes` and `body.end()`; we
    // don't want to walk off the end, but we also don't want to copy too much
    // into `buffer`.
    auto endPtr = std::min(iter + bytes, end);
    // std::copy returns one past the end of the copy.
    auto bufferEnd = std::copy(iter, (decltype(iter))endPtr, buffer);
    // The number of bytes copied is exactly the distance from `buffer` to
    // `bufferEnd`.
    size_t numBytesCopied = std::distance(buffer, bufferEnd);
    // Increment bodyIter by the number of bytes copied.
    iter += numBytesCopied;
    return numBytesCopied;
  }
};

/// Class template argument deduction guide to suppress warnings.
template <typename T>
ContainerReadCallbackAdaptor(T &) -> ContainerReadCallbackAdaptor<T>;

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
    TimeoutError,      // Response exceeded timeout.
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
/// Thread safety: HTTPClient is thread safe but doesn't provide
/// synchronization.
class HTTPClient {
public:
  HTTPClient(HTTPContextRef ctx);
  virtual ~HTTPClient();

  /// Blocking call that executes the HTTPRequest and writes the response to the
  /// provided ostream. Returns a HTTPResponse.
  ///
  /// Request `timeout` and `maxLength` can specified to limit requests.
  /// A `timeout` and `maxLength` of zero will not limit the request.
  HTTPResponse executeRequest(
      const HTTPRequest &request, raw_ostream &os,
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
      size_t maxLength = 0);

protected:
  /// Core implementation of executeRequest. This includes all the calls to
  /// libcurl. The separation is so the base class can implement any state
  /// checking that needs to happen before the actual network request, while
  /// allowing subclasses to implement the network request any way they like.
  virtual HTTPResponse executeRequestImpl(
      const HTTPRequest &request, raw_ostream &os,
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
      size_t maxLength = 0);

private:
  HTTPContextRef context;
  void *curl = nullptr;
};
} // namespace M

#endif // SUPPORT_HTTP_HTTPCLIENT_H
