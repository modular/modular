//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MODULAR_COMMON_HTTPCLIENT_H
#define MODULAR_COMMON_HTTPCLIENT_H

#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/ErrorOr.h"
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
  std::string URL;
  bool verify_tls_peer = true;
};

/// HTTPClient that wraps libcurl.
///
/// Thead safety: HTTPClient is thread safe but doesn't provide synchronization.
class HTTPClient {
public:
  HTTPClient(HTTPContextRef ctx);
  ~HTTPClient();

  /// Blocking call that executes the HTTPRequest and writes the response to the
  /// provided ostream.
  ///
  /// Request `timeout` and `maxLength` can specified to limit requests.
  /// A `timeout` and `maxLength` of zero will not limit the request.
  ErrorOrSuccess executeRequest(
      const HTTPRequest &request, raw_ostream &os,
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
      size_t maxLength = 0);

private:
  HTTPContextRef context;
  void *curl = nullptr;
};

} // namespace M

#endif // MODULAR_COMMON_HTTPCLIENT_H
