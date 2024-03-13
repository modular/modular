//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/Backends/HTTPCacheBackend.h"
#include "Support/Base64.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace Cache;
using namespace std::chrono_literals;

ErrorOrSuccess HTTPCacheBackend::insertImpl(StringRef keyHash, BufferRef obj) {
  return Error::getStaticString("HTTP backend does not support insert");
}

ErrorOr<bool> HTTPCacheBackend::containsImpl(StringRef keyHash) const {
  auto findOr = requestImpl(keyHash, /*headOnly=*/true);
  if (findOr.isError())
    return findOr.takeError();

  return findOr->has_value();
}

ErrorOr<std::optional<BufferRef>>
HTTPCacheBackend::findImpl(StringRef keyHash) const {
  return requestImpl(keyHash, /*headOnly=*/false);
}

ErrorOr<std::optional<BufferRef>>
HTTPCacheBackend::requestImpl(StringRef keyHash, bool headOnly) const {
  // Base64 encode the key hash so it's URL-safe.
  std::string keyHashB64 = encodeURLSafeBase64(keyHash);

  // Maximum of 512M per request.
  constexpr size_t maxBytes = 1024 * 1024 * 512;

  // We didn't have it locally, so create a request to go get it.
  HTTPRequest req{/*URL=*/url + "/" + keyHashB64};
  req.progress = progress; // Pass through the progress meter.
  if (headOnly)
    req.method = HTTPRequest::HEAD;

  // Either create a new WriteableBuffer or use the one that was passed in.
  // Note that if this is a head-only request, we still reserve the space for
  // the response (we just set a maximum size of zero bytes).
  auto writeBuf = WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                       /*capacity=*/headOnly ? 0 : maxBytes);

  int retryCount = 0;
  while (true) {
    HTTPClient client(ctx.copy());
    // No auth needed for now.
    client.noAuthNeeded();

    // Execute the request. This is a blocking request on this thread.
    HTTPResponse response = client.executeRequest(
        req, *writeBuf, std::chrono::milliseconds::zero(), maxBytes);

    // TODO: Will the result bytes be encoded or can we expect them to be raw
    //       binary?

    // Everything is fine, return the buffer.
    if (response.isSuccess()) {
      return std::move(writeBuf);
    }

    // Content was not found - this is not an error, just return nullopt.
    if (response.isError() && response.responseCode &&
        *response.responseCode == HTTPResponseCode::NotFound)
      return std::nullopt;

    if (response.isError() &&
        (response.kind == HTTPResponse::Kind::TimeoutError ||
         response.kind == HTTPResponse::Kind::TransportError ||
         (response.responseCode && *response.responseCode >= 500 &&
          *response.responseCode < 600))) {

      // Retry on timeout, transport error, or temporary server error up to 3
      // times.
      if (retryCount++ < 3) {
        // Reset our write buffer in case we had some junk in there.
        writeBuf = WriteableBuffer::get(/*size=*/0, /*alignment=*/std::nullopt,
                                        /*capacity=*/headOnly ? 0 : maxBytes);
        // Exponential backoff to wait until things are hopefully working again.
        // Sleep for 2^retryCount seconds before retrying.
        std::this_thread::sleep_for((1 << retryCount) * 1000ms);
        continue;
      }
    }

    // Every other kind of error is not recoverable so don't even try.
    // Return the error we hit.
    std::string errorContextStr =
        llvm::formatv("Looking for {0}", keyHashB64).str();
    return response.asError(errorContextStr).takeError();
  }
}

HTTPCacheBackendRef M::Cache::getHTTPCacheBackend(HTTPContextRef ctx,
                                                  std::string url,
                                                  Runtime &runtime,
                                                  Progress *progress) {
  return HTTPCacheBackendRef::create(std::move(ctx), std::move(url), runtime,
                                     progress);
}
