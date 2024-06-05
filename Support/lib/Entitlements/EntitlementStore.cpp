//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Entitlements/EntitlementStore.h"
#include "Support/Base64.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/Cryptography/Keypair.h"
#include "Support/Entitlements/EntitlementToken.h"
#include "Support/FileSystemExtras.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/Random.h"
#include "mbedtls/error.h"
#include "mbedtls/pem.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/x509_crt.h"
#include "mbedtls/x509_csr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "RootCert.inc"

#ifdef __APPLE__
#include <Security/Security.h>
#endif // __APPLE__

#ifdef _WIN32
#include <windows.h>

#include <wincrypt.h>
#else
#include <unistd.h>
#endif // _WIN32

#include <chrono>
#include <filesystem>

using namespace M;

//===----------------------------------------------------------------------===//
// mbedTLS Extension Callback
//===----------------------------------------------------------------------===//

namespace {
struct MbedTLSCallbackContext {
  llvm::DenseMap<ASN1::ObjectID, std::unique_ptr<Entitlement>> &store;
  std::optional<Error> error;
};
} // namespace

static constexpr int oidDecodingError = -1;
static constexpr int nonModularOID = -2;
static constexpr int entitlementParsingError = -3;

/// Parses the extension provided and if it's a modular entitlement OID, then it
/// parses it and places it in the map.
static int extensionCallback(void *context, mbedtls_x509_crt const *crt,
                             mbedtls_x509_buf *oidBuf, int critical,
                             const unsigned char *dataBegin,
                             const unsigned char *dataEnd) {
  auto *ctx = (MbedTLSCallbackContext *)context;

  auto oidOr =
      ASN1::ObjectID::fromEncoded(ArrayRef<uint8_t>(oidBuf->p, oidBuf->len));
  if (oidOr.isError()) {
    ctx->error = oidOr.takeError();
    return oidDecodingError;
  }

  ASN1::ObjectID oid = std::move(*oidOr);

  // We can't handle non-Modular OIDs
  if (!oid.isModularOID())
    return nonModularOID;

  auto entitlementOr = Entitlement::parse(
      oid, (bool)critical, ArrayRef<uint8_t>(dataBegin, dataEnd));
  if (entitlementOr.isError()) {
    ctx->error = entitlementOr.takeError();
    return entitlementParsingError;
  }

  // Add the entitlement. We use `oid` rather than the entitlement's OID because
  // in case the entitlement is unknown, we don't want to simply drop it on the
  // floor. This will at least save the OID, even if we don't actually parse the
  // data or anything.
  ctx->store[oid] = std::move(*entitlementOr);
  // Success, return 0.
  return 0;
}

//===----------------------------------------------------------------------===//
// mbedTLSErrorToString
//===----------------------------------------------------------------------===//

/// Convert an mbedTLS error to a human-readable M::Error object. For non-debug
/// builds, this generates generic error codes.
static std::string mbedTLSErrorToString(int rc) {
#ifdef MODULAR_DEBUG
  std::string errStr(1024, '\0');
  mbedtls_strerror(rc, errStr.data(), errStr.size());
  // Use strlen because mbedtls_strerror guarantees a null-terminated string, so
  // we can use this to determine the actual length.
  errStr.resize(strlen(errStr.c_str()));
  errStr.shrink_to_fit();
  return errStr;
#else  // MODULAR_DEBUG
  return "mbedTLS encountered an error, aborting";
#endif // MODULAR_DEBUG
}

//===----------------------------------------------------------------------===//
// csprng
//===----------------------------------------------------------------------===//

/// Provide the platform-specific csprng call. This is an mbedTLS-compatible
/// adaptor to SecureRandomBytesGenerator.
static int csprng(void *ctx, unsigned char *buf, size_t numBytes) {
  auto *rng = (SecureRandomBytesGenerator *)ctx;
  MutableArrayRef<uint8_t> randBuf(buf, numBytes);
  if (auto err = rng->getRandomBytes(randBuf))
    return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// getSystemRootCerts
//===----------------------------------------------------------------------===//

/// Get the root certificates for this system - parse them and add them to
/// `list`.
static ErrorOrSuccess getSystemRootCerts(mbedtls_x509_crt *list) {
  // The first cert in the list is going to be our stored certificate. Unless
  // something highly unforseen happens, that will be the root cert that we want
  // to trust.
  int rc = mbedtls_x509_crt_parse(list, modularRootCertificate.bytes_begin(),
                                  modularRootCertificate.size() + 1);
  if (rc != 0) {
    return Error("could not parse the modular root certificate: " +
                 mbedTLSErrorToString(rc));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain
//===----------------------------------------------------------------------===//

/// This class provides a C++ wrapper around an mbedTLS certificate chain. It
/// encapsulates the basic certificate operations like parsing from PEM and
/// verifying the chain of trust based on the system root certificates.
namespace M::Detail {
class CertificateChain {
public:
  CertificateChain() { mbedtls_x509_crt_init(&parsed); }
  ~CertificateChain() { mbedtls_x509_crt_free(&parsed); }

  /// Parse a CertificateChain from a PEM buffer. Apply the callback `cb` and
  /// `ctx` to each certificate in the chain as they're being parsed. The buffer
  /// `pem` may contain more than one certificate.
  static ErrorOr<std::unique_ptr<CertificateChain>>
  fromPEM(mbedtls_x509_crt_ext_cb_t cb, void *ctx, BufferRef pem);

  /// Verify that the certificate chain is valid based on the system root certs,
  /// and that the public key is the correct pairing for `clientKeys`. If a CRL
  /// is provided (in PEM format) then it's used to check if any certificates in
  /// the chain have been revoked. The CRL PEM must include the null terminating
  /// byte at the end!
  ErrorOrSuccess verify(Keypair &clientKeys);

  /// Get the PEM buffer for this certificate chain.
  StringRef getPEM() const {
    assert(verified && "must have a verified certificate chain");
    return pem->getBuffer();
  }

  /// Check if the certificate chain is available for use. Note that this does
  /// NOT necesarily mean that it's been verified!
  bool isAvailable() const {
    return bool(pem) && pem->getBufferSize() != 0 && parsed.version != 0;
  }

  /// Return the subject of the leaf certificate. This asserts that the
  /// certificate has been verified. This returns a reference to the string data
  /// contained directly in the certificate.
  ErrorOr<Detail::CertSubject> getSubject() const;

  /// Apply `policy` to the validity period of the parsed leaf certificate.
  ErrorOr<bool> applyToValidity(
      llvm::function_ref<bool(std::chrono::system_clock::time_point,
                              std::chrono::system_clock::time_point)>
          policy);

private:
  /// Get the leaf certificate in the chain. This is useful because that will be
  /// the client's cert - any intermediate CAs will be higher in the chain than
  /// the actual client cert.
  const mbedtls_x509_crt *getLeafCertificate() const;

  /// Store the PEM-encoded bytes of the client certificate chain here to be
  /// flushed if necessary/requested.
  BufferRef pem;

  /// Store the parsed client certificate chain as well. This is absolutely
  /// redundant with the PEM representation above, but parsing a chain of
  /// certificates is non-trivial and having them already-parsed is valuable.
  mbedtls_x509_crt parsed = {};

  /// True if and only if the client has called `verify`. Note that this does
  /// NOT carry over across application shutdown for any reason - the
  /// certificates must be re-verified every time we start up.
  bool verified = false;
};
} // namespace M::Detail

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::fromPEM
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<Detail::CertificateChain>>
Detail::CertificateChain::fromPEM(mbedtls_x509_crt_ext_cb_t cb, void *ctx,
                                  BufferRef pem) {
  auto out = std::make_unique<CertificateChain>();
  out->pem = std::move(pem);
  // Set up the PEM context.
  mbedtls_pem_context pemCtx;
  mbedtls_pem_init(&pemCtx);

  // Parse the chain of certificates. These will be concatenated as PEM
  // buffers one after the other, so we simply parse until we run out of data.
  size_t bytesConsumed = 0;
  for (; bytesConsumed < out->pem->getBufferSize();) {
    // Parse a single certificate first from PEM, then from DER.
    // Note this method frees ctx on error before returning.
    size_t bytes = 0;
    int rc = mbedtls_pem_read_buffer(
        &pemCtx, "-----BEGIN CERTIFICATE-----", "-----END CERTIFICATE-----",
        (const uint8_t *)out->pem->getBufferStart() + bytesConsumed,
        /*pwd=*/nullptr,
        /*pwdlen=*/0, &bytes);
    if (rc != 0)
      return Error(mbedTLSErrorToString(rc));

    bytesConsumed += bytes;

    size_t buflen = 0;
    const uint8_t *derBuf = mbedtls_pem_get_buffer(&pemCtx, &buflen);
    // Parse the client cert into the cert chain. This will also populate the
    // store with the entitlements found in the certificate. We do need to
    // perform a copy for any data that was previously in PEM form, because
    // otherwise the buffer will be freed when we free the PEM buffer.
    rc = mbedtls_x509_crt_parse_der_with_ext_cb(&out->parsed, derBuf, buflen,
                                                /*make_copy=*/1,
                                                /*cb=*/cb, /*p_ctx=*/ctx);

    // Free the pem memory as we will either alloc a new chunk in the next
    // iteration or return out.  Otherwise we will leak it.
    mbedtls_pem_free(&pemCtx);

    if (rc != 0)
      return Error(mbedTLSErrorToString(rc));
  }

  return out;
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::verify
//===----------------------------------------------------------------------===//

ErrorOrSuccess Detail::CertificateChain::verify(Keypair &clientKeys) {
  // Find and open the root certs.
  mbedtls_x509_crt caCerts;
  mbedtls_x509_crt_init(&caCerts);
  auto freeCACerts =
      llvm::make_scope_exit([&] { mbedtls_x509_crt_free(&caCerts); });
  if (auto err = getSystemRootCerts(&caCerts))
    return err.takeError();

  mbedtls_x509_crl caCRL;
  mbedtls_x509_crl_init(&caCRL);
  auto freeCRL = llvm::make_scope_exit([&] { mbedtls_x509_crl_free(&caCRL); });

  uint32_t flags = 0;
  // We use the expected next profile mbedTLS provides, but we disallow
  // RSA-2048. This profile targets a 128-bit security level.
  mbedtls_x509_crt_profile profile = mbedtls_x509_crt_profile_next;
  profile.rsa_min_bitlen = 3072;

  // Verify the certificate with the provided security profile. We don't
  // provide additional callbacks, and we don't really care about the common
  // name.
  int rc = mbedtls_x509_crt_verify_with_profile(
      &parsed, &caCerts, &caCRL, &profile,
      /*cn=*/nullptr, &flags, /*f_vrfy=*/nullptr, /*p_vrfy=*/nullptr);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  const mbedtls_x509_crt *leafCert = getLeafCertificate();

  // OK - we now have/can get the public key. Ensure the cert was signed by
  // *this* private key. We do this by checking that the public key on the cert
  // matches the private key we just parsed.
  SecureRandomBytesGenerator rng;
  rc = mbedtls_pk_check_pair(&leafCert->pk, clientKeys.getRawKey(), &csprng,
                             &rng);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  verified = true;
  return success();
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::getSubject
//===----------------------------------------------------------------------===//

ErrorOr<Detail::CertSubject> Detail::CertificateChain::getSubject() const {
  assert(verified && "must have a verified certificate chain");
  // The subject is exactly the value of the ASN.1 subject for the previous
  // certificate, which is the user ID. The subject is a linked-list of C/O/CN
  // so we have to find the common name, which is identified by OID 2.5.4.3.
  auto cnOr = ASN1::ObjectID::fromString("2.5.4.3");
  if (cnOr.isError()) {
    // Error resolving OID. Should never happen since 2.5.4.3 is a legit OID.
    return cnOr.takeError();
  }
  SmallVector<uint8_t> encoded = cnOr->getEncoded();

  auto *commonName = mbedtls_asn1_find_named_data(
      &getLeafCertificate()->subject, (const char *)encoded.data(),
      encoded.size());
  if (commonName == nullptr) {
    // Did not find CN!
    return Error("CN missing from cert");
  }

  auto CN = StringRef{(const char *)commonName->val.p, commonName->val.len};

  // Look for OU
  auto ouOr = ASN1::ObjectID::fromString("2.5.4.11");
  if (ouOr.isError()) {
    // Error resolving OID. Should never happen since 2.5.4.11 is a legit OID.
    // If it does fail for some reason, return a valid-enough Subject with OU=CN
    return Detail::CertSubject{CN.str(), CN.str()};
  }
  SmallVector<uint8_t> encodedOU = ouOr->getEncoded();

  auto *ou = mbedtls_asn1_find_named_data(&getLeafCertificate()->subject,
                                          (const char *)encodedOU.data(),
                                          encodedOU.size());
  if (ou == nullptr) {
    // If OU is not defined, backfill OU=CommonName
    return Detail::CertSubject{CN.str(), CN.str()};
  }
  auto OU = StringRef{(const char *)ou->val.p, ou->val.len};
  return Detail::CertSubject{CN.str(), OU.str()};
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::applyToValidity
//===----------------------------------------------------------------------===//

ErrorOr<bool> Detail::CertificateChain::applyToValidity(
    llvm::function_ref<bool(std::chrono::system_clock::time_point,
                            std::chrono::system_clock::time_point)>
        policy) {
  const mbedtls_x509_crt *leaf = getLeafCertificate();
  tm validFrom = {/*tm_sec=*/leaf->valid_from.sec,
                  /*tm_min=*/leaf->valid_from.min,
                  /*tm_hour=*/leaf->valid_from.hour,
                  /*tm_mday=*/leaf->valid_from.day,
                  /*tm_mon*/ leaf->valid_from.mon - 1,
                  /*tm_year=*/leaf->valid_from.year - 1900,
                  /*tm_wday=*/-1,
                  /*tm_yday=*/-1,
#ifndef _WIN32
                  /*tm_isdst=*/-1,
                  /*tm_gmtoff=*/-1,
                  /*tm_zone=*/nullptr
#else  // _WIN32
                  /*tm_isdst=*/0 // Windows' gmtime always returns 0 for this.
#endif // _WIN32
  };
  time_t normalizedFrom = mktime(&validFrom);
  if (normalizedFrom == (time_t)-1)
    return Error("invalid validFrom date in certificate");

  tm validTo = {/*tm_sec=*/leaf->valid_to.sec,
                /*tm_min=*/leaf->valid_to.min,
                /*tm_hour=*/leaf->valid_to.hour,
                /*tm_mday=*/leaf->valid_to.day,
                /*tm_mon*/ leaf->valid_to.mon - 1,
                /*tm_year=*/leaf->valid_to.year - 1900,
                /*tm_wday=*/-1,
                /*tm_yday=*/-1,
#ifndef _WIN32
                /*tm_isdst=*/-1,
                /*tm_gmtoff=*/-1,
                /*tm_zone=*/nullptr
#else  // _WIN32
                /*tm_isdst=*/0 // Windows' gmtime always returns 0 for this.
#endif // _WIN32
  };
  time_t normalizedTo = mktime(&validTo);
  if (normalizedTo == (time_t)-1)
    return Error("invalid validTo date in certificate");

  return policy(std::chrono::system_clock::from_time_t(normalizedFrom),
                std::chrono::system_clock::from_time_t(normalizedTo));
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::getLeafCertificate
//===----------------------------------------------------------------------===//

const mbedtls_x509_crt *Detail::CertificateChain::getLeafCertificate() const {
  // The client cert is the leaf certificate, so grab it from the
  // start of the chain.
  return &parsed;
}

//===----------------------------------------------------------------------===//
// M::defaultEntitlementRefreshPolicy
//===----------------------------------------------------------------------===//

bool M::defaultEntitlementRefreshPolicy(
    std::chrono::system_clock::time_point from,
    std::chrono::system_clock::time_point to) {
  // std::duration with some complicated template parameters, hence using auto.
  auto midpointDur = (to - from) / 2;
  std::chrono::system_clock::time_point midpoint = from + midpointDur;
  return std::chrono::system_clock::now() > midpoint;
}

//===----------------------------------------------------------------------===//
// requestDeviceCode
//===----------------------------------------------------------------------===//

/// Request the device code from Phoenix. This allows us to initiate the Device
/// Authorization Flow by requesting the device code from the server.
static ErrorOr<llvm::json::Value> requestDeviceCode(HTTPClient &client) {
  // Call the GetDeviceCode endpoint. This will return a JSON blob with the URL
  // the user needs to visit.
  HTTPRequest request = {
      modularAuthURL + "/v1/oauth/device/authorize",
  };
  request.method = HTTPRequest::Method::POST;
  request.headers.try_emplace("content-type", "application/json");

  llvm::StringLiteral body = R"({
    "audience": "",
    "client_id": "mcl_XrMVVoDY8fK6NjpodUeXLlH2LFiZn5Ji",
    "scope": "openid"
  })";
  request.bodyLen = body.size();
  request.body = ContainerReadCallbackAdaptor(body);

  // Get the JSON object back.
  size_t maxSize = 1024;
  WriteableBufferRef responseBuf = WriteableBuffer::get();

  // Execute the request!
  auto response = client.executeRequest(
      request, *responseBuf, std::chrono::milliseconds::zero(), maxSize);
  if (response.isError())
    return response.asError().takeError();

  // Parse the JSON looking for the device code and the
  // verification_uri_complete to give to the user.
  auto jsonOr = llvm::json::parse(responseBuf->Buffer::getBuffer());
  if (!jsonOr)
    return Error(llvm::toString(jsonOr.takeError()));

  return std::move(*jsonOr);
}

//===----------------------------------------------------------------------===//
// pollForOAuthTokens
//===----------------------------------------------------------------------===//

/// Given a device code, we can now poll for the OAuth tokens. This takes an
/// interval and the device code returned from `requestDeviceCode`. This
/// function may take a long time to complete, as it polls for at most 256 *
/// `interval` seconds. Note also that the server may respond with "slow_down",
/// which will cause us to increment `interval` by 5 seconds each time.
static ErrorOr<std::string> pollForOAuthTokens(HTTPClient &client,
                                               std::chrono::seconds interval,
                                               llvm::StringRef deviceCode) {
  // Now poll `modularAuthURL`/v1/oauth/token for the token from the
  // user.
  HTTPRequest pollRequest = {modularAuthURL + "/v1/oauth/token"};
  pollRequest.method = HTTPRequest::Method::POST;
  pollRequest.headers.try_emplace("content-type", "application/json");

  llvm::StringLiteral bodyFmtStr = R"({
    "client_id": "mcl_XrMVVoDY8fK6NjpodUeXLlH2LFiZn5Ji",
    "device_code": "{0}",
    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
  })";
  std::string pollRequestBody = llvm::formatv(bodyFmtStr.data(), deviceCode);

  pollRequest.bodyLen = pollRequestBody.size();

  // Poll for 256 iterations, or until the token is non-empty.
  std::string accessToken;
  int maxIters = 256;
  size_t maxSize = 1024;
  // TODO: https://datatracker.ietf.org/doc/html/rfc8628#section-3.5 says we
  //       should unilaterally implement a backoff mechanism if we hit a
  //       timeout.
  while (accessToken.empty() && maxIters-- > 0) {
    // Set the body on the request - we have to do this every time because the
    // iterators get advanced while we do the read.
    pollRequest.body = ContainerReadCallbackAdaptor(pollRequestBody);

    // Do the request.
    WriteableBufferRef pollResponseBuf = WriteableBuffer::get();
    auto pollResponse =
        client.executeRequest(pollRequest, *pollResponseBuf,
                              std::chrono::milliseconds::zero(), maxSize);

    auto responseJSONOr =
        llvm::json::parse(pollResponseBuf->Buffer::getBuffer());
    if (!responseJSONOr)
      return Error(llvm::toString(responseJSONOr.takeError()));

    llvm::json::Object *responseJSON = responseJSONOr->getAsObject();
    if (!responseJSON)
      return Error("/oauth/token invalid response, expected JSON object");

    if (pollResponse.isError()) {
      // If it was a bad request, then we might still be waiting on the
      // authorization or something. Parse those cases out.
      if (pollResponse.responseCode == HTTPResponseCode::BadRequest) {
        auto errOr = responseJSON->getString("error");
        if (!errOr) {
          return Error("/oauth/token invalid response, expected 'error' in "
                       "response payload");
        }

        // If we're still waiting on the authorization, then sleep this thread
        // for some time and re-enter the loop.
        if (*errOr == "authorization_pending") {
          std::this_thread::sleep_for(interval);
          continue;
        } else if (*errOr == "slow_down") {
          interval += std::chrono::seconds(5);
          std::this_thread::sleep_for(interval);
          continue;
        }

        // Fallthrough to returning the response as an error.
      }
      return pollResponse.asError().takeError();
    }

    // Pull out the access token.
    auto accessTokOr = responseJSON->getString("access_token");
    if (!accessTokOr) {
      return Error("/oauth/token invalid response, expected 'access_token' in "
                   "response payload");
    }
    accessToken = *accessTokOr;
  }

  // If the token is still empty, return.
  if (accessToken.empty()) {
    return Error(
        "max number of requests reached (256) to /oauth/token endpoint");
  }

  return accessToken;
}

//===----------------------------------------------------------------------===//
// authAndFetchToken
//===----------------------------------------------------------------------===//

/// This function uses the OAuth Device Authorization Flow to do initial
/// authentication and bootstrap the client certificate.
static ErrorOr<std::string> authAndFetchToken(HTTPClient &client,
                                              bool openBrowser) {
  auto jsonOr = requestDeviceCode(client);
  if (jsonOr.isError())
    return jsonOr.takeError();

  llvm::json::Object *jsonResponse = jsonOr->getAsObject();
  if (!jsonResponse)
    return Error("/oauth/device/code invalid response, expected JSON object");

  auto deviceCodeOr = jsonResponse->getString("device_code");
  if (!deviceCodeOr) {
    return Error(
        "/oauth/device/code invalid response, expected 'device_code' in "
        "response payload");
  }

  auto userCodeOr = jsonResponse->getString("user_code");
  if (!userCodeOr) {
    return Error("/oauth/device/code invalid response, expected 'user_code' in "
                 "response payload");
  }

  auto intervalOr = jsonResponse->getInteger("interval");
  if (!intervalOr) {
    return Error("/oauth/device/code invalid response, expected 'interval' in "
                 "response payload");
  }

  auto verifURIOr = jsonResponse->getString("verification_uri_complete");
  if (!verifURIOr) {
    return Error("/oauth/device/code invalid response, expected "
                 "'verification_uri_complete' in response payload");
  }

  llvm::outs() << "To complete auth, open this web page:\n"
               << *verifURIOr << "\n\n"
               << "Verify using this code:\n"
               << *userCodeOr << "\n\n";
#if defined(_MSC_VER)
  const std::string program = "start";
  const bool startAllowed = true;
#elif defined(__APPLE__)
  const std::string program = "open";
  const bool startAllowed = geteuid() != 0;
#else
  const std::string program = "xdg-open";
  // Let's make sure the Linux host has a display set and available.
  bool displayAvailable = false;
  const StringRef displayVars[3] = {"XDG_SESSION_DESKTOP", "DISPLAY",
                                    "WAYLAND_DISPLAY"};
  for (auto displayVar : displayVars)
    displayAvailable =
        displayAvailable || llvm::sys::Process::GetEnv(displayVar);
  const bool startAllowed = geteuid() != 0 && displayAvailable;
#endif
  if (startAllowed && openBrowser) {
    if (auto fullProgramOr = llvm::sys::findProgramByName(program)) {
      SmallVector<StringRef> argVec = {program, *verifURIOr};
      std::string errMsg;
      int result = llvm::sys::ExecuteAndWait(
          *fullProgramOr, argVec, std::nullopt, /*Redirects=*/{"", "", ""},
          /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errMsg);
      if (result == 0)
        llvm::outs() << "A browser should be opened automatically.\n";
      else {
        llvm::errs() << "Failed to automatically open browser: " + errMsg +
                            "\n";
        llvm::outs() << "Please open the above link in a web browser.\n";
      }
    }
  }
  llvm::outs() << "Waiting for confirmation...\n";

  return pollForOAuthTokens(client, std::chrono::seconds(*intervalOr),
                            *deviceCodeOr);
}

static ErrorOr<llvm::json::Value>
requestUserInfo(HTTPClient &client, std::optional<std::string> accessToken) {
  HTTPRequest request = {
      modularAuthURL + "/v1/oidc/userinfo",
  };
  request.method = HTTPRequest::Method::GET;
  request.headers.try_emplace("content-type", "application/json");
  if (accessToken)
    request.accessToken.emplace(std::move(*accessToken));

  // Get the JSON object back.
  size_t maxSize = 1024;
  WriteableBufferRef responseBuf = WriteableBuffer::get();

  // Execute the request!
  auto response = client.executeRequest(
      request, *responseBuf, std::chrono::milliseconds::zero(), maxSize);
  if (response.isError())
    return response.asError().takeError();

  // Parse the JSON looking for the device code and the
  // verification_uri_complete to give to the user.
  auto jsonOr = llvm::json::parse(responseBuf->Buffer::getBuffer());
  if (!jsonOr)
    return Error(llvm::toString(jsonOr.takeError()));

  return std::move(*jsonOr);
}

//===----------------------------------------------------------------------===//
// generateCSR
//===----------------------------------------------------------------------===//

/// Given a keypair and a subject generate a CSR that would generate a
/// certificate of the kind we expect to receive. This generates the data in PEM
/// format because it's always sent directly over the wire. The entitlements are
/// added by the service, so we don't have to fetch the list.
static ErrorOrSuccess generateCSR(Keypair &keys,
                                  const Detail::CertSubject &subject,
                                  const WriteableBufferRef &buf) {
  mbedtls_x509write_csr csr;
  mbedtls_x509write_csr_init(&csr);
  auto freeCSR =
      llvm::make_scope_exit([&] { mbedtls_x509write_csr_free(&csr); });

  // Set the subject name fields.
  int rc =
      mbedtls_x509write_csr_set_subject_name(&csr, subject.format().c_str());
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Set the key to the keypair we either just generated or just found.
  mbedtls_x509write_csr_set_key(&csr, keys.getRawKey());

  // Use SHA-256 for signatures.
  mbedtls_x509write_csr_set_md_alg(&csr, MBEDTLS_MD_SHA256);

  // We need to be able to use this key for signing, non-repudiation, encrypting
  // keys, and agreeing on a session key.
  rc = mbedtls_x509write_csr_set_key_usage(
      &csr,
      MBEDTLS_X509_KU_DIGITAL_SIGNATURE | MBEDTLS_X509_KU_NON_REPUDIATION |
          MBEDTLS_X509_KU_KEY_ENCIPHERMENT | MBEDTLS_X509_KU_KEY_AGREEMENT);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // See
  // https://forums.mbed.com/t/does-mbedtls-has-api-to-get-the-size-of-csr-data/5293/7
  // - apparently generally speaking, a 2K buffer is enough.
  std::array<uint8_t, 2048> tmp = {};

  // Generate the DER buffer.
  SecureRandomBytesGenerator rng;
  rc = mbedtls_x509write_csr_pem(&csr, tmp.data(), tmp.size(), &csprng, &rng);
  if (rc < 0)
    return Error(mbedTLSErrorToString(rc));

  // Copy the PEM bytes into the WritableBufferRef we just got passed.
  auto *tmpPtr = (const char *)tmp.data();
  buf->write(tmpPtr, strlen(tmpPtr));

  // Done.
  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore Constructor/Destructor
//===----------------------------------------------------------------------===//

EntitlementStore::EntitlementStore(Keypair &&clientKeys,
                                   BufferRef &&clientKeyPriv,
                                   BufferRef &&clientCert)
    : clientKeys(std::move(clientKeys)),
      clientKeyPriv(std::move(clientKeyPriv)),
      clientCert(std::move(clientCert)) {}
EntitlementStore::EntitlementStore(M::EntitlementStore &&other) = default;
EntitlementStore::~EntitlementStore() = default;

//===----------------------------------------------------------------------===//
// EntitlementStore::create
//===----------------------------------------------------------------------===//

ErrorOr<EntitlementStore> EntitlementStore::create(BufferRef &&clientKeyPriv,
                                                   BufferRef &&clientCert) {
  // Open the private key.
  auto privKeyOr = Keypair::open(clientKeyPriv);
  if (privKeyOr.isError())
    return privKeyOr.takeError();

  // Parse all entitlements.
  EntitlementStore out = EntitlementStore(
      std::move(*privKeyOr), std::move(clientKeyPriv), std::move(clientCert));

  // Parse the certificate chain. This is validating the store. The clientCert
  // object will be stored by this and all entitlements updated.
  if (auto err = out.parseCertificateChain())
    return err.takeError();

  return out;
}

//===----------------------------------------------------------------------===//
// EntitlementStore::getUserID
//===----------------------------------------------------------------------===//

ErrorOr<std::string> EntitlementStore::getUserID() const {
  if (!clientCertChain || !clientCertChain->isAvailable())
    return Error("no client certificate");

  auto subjOr = clientCertChain->getSubject();
  if (subjOr.isError())
    return subjOr.takeError();

  return subjOr->UserId;
}

//===----------------------------------------------------------------------===//
// EntitlementStore::open
//===----------------------------------------------------------------------===//

static std::filesystem::path findClientFile(const std::filesystem::path &dir,
                                            StringRef cfgVal, StringRef name) {
  // Use the configuration file.
  if (!cfgVal.empty())
    return std::filesystem::path(std::string(cfgVal));
  // Try to find the file if it exists.
  auto fileOr = findModularFile(name);
  if (fileOr)
    return std::filesystem::path(std::string(*fileOr));
  // Assume it is a file in the given directory.
  return dir / std::filesystem::path(std::string(name));
}

ErrorOr<std::optional<EntitlementStore>>
EntitlementStore::open(Config &config, std::optional<std::string> envVarOr) {
  // Use the creds in the EntitlementToken if:
  //   - The containing env var is defined
  //   - The value successfully unpacks to an EntitlementToken
  //   - At least 1 cert in certChain
  // Otherwise fallback to looking at the file system for certs.

  if (envVarOr.has_value()) {
    if (auto tokenOr = unpackToken(envVarOr.value()); !tokenOr.isError()) {
      const EntitlementToken token = *tokenOr.takeValue().get();
      if (auto storeOr = fromToken(token); !storeOr.isError()) {
        return storeOr;
      }
    }
  }

  return fromConfig(config);
}

ErrorOr<EntitlementStore>
EntitlementStore::fromToken(const EntitlementToken &token) {
  if (token.certChain.empty()) {
    return Error("Empty certChain");
  }
  BufferRef tokenBuf = Buffer::get(token.key);

  // Concatenate all the certs together
  WriteableBufferRef certBuf = WriteableBuffer::get();
  for (const auto &pem : token.certChain) {
    *certBuf << pem;
    // Make sure that each cert starts on a newline
    if (pem.back() != '\n') {
      *certBuf << "\n";
    }
  }

  return create(std::move(tokenBuf), std::move(certBuf));
}

ErrorOr<EntitlementStore> EntitlementStore::fromConfig(Config &config) {
  // Key files and certificate paths.
  auto dataFolderOr = M::Config::getModularDataFolderPath(true);
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();
  auto clientKeyPrivPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_key_priv"),
      "client_priv.pem");
  auto clientCertPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_cert"), "client.pem");

  // Attempt to load the buffers.
  auto clientKeyPrivOr = Buffer::getFile(clientKeyPrivPath);
  if (clientKeyPrivOr.isError())
    return clientKeyPrivOr.takeError();
  auto clientCertOr = Buffer::getFile(clientCertPath);
  if (clientCertOr.isError())
    return clientCertOr.takeError();

  // Return the initialized store.
  return create(std::move(*clientKeyPrivOr), std::move(*clientCertOr));
}

//===----------------------------------------------------------------------===//
// JWT
//===----------------------------------------------------------------------===//

/// This class provides a read adaptor for JWTs. This will parse the token,
/// decoding it and making the fields available as raw JSON.
///
/// WARNING: This does NOT verify the token in any way! Only trust tokens that
///          come from trusted sources!
namespace {
class JWT {
public:
  static ErrorOr<JWT> parse(StringRef str) {
    SmallVector<StringRef, 3> parts;
    str.split(parts, '.');

    JWT out;

    // Parse the header.
    auto decodedOr = decodeURLSafeBase64(parts[0]);
    if (decodedOr.isError())
      return decodedOr.takeError();
    out.headerStr = std::move(*decodedOr);

    auto headerOr = llvm::json::parse(out.headerStr);
    if (!headerOr)
      return Error(llvm::toString(headerOr.takeError()));
    out.headerVal = std::move(*headerOr);

    if (!out.headerVal.getAsObject())
      return Error("invalid JWT - expected a JSON object header");

    // Parse the payload
    decodedOr = decodeURLSafeBase64(parts[1]);
    if (decodedOr.isError())
      return decodedOr.takeError();
    out.payloadStr = std::move(*decodedOr);

    auto payloadOr = llvm::json::parse(out.payloadStr);
    if (!payloadOr)
      return Error(llvm::toString(payloadOr.takeError()));
    out.payloadVal = std::move(*payloadOr);

    if (!out.payloadVal.getAsObject())
      return Error("invalid JWT - expected a JSON object payload");

    return std::move(out);
  }

  llvm::json::Object *getHeader() { return headerVal.getAsObject(); }
  llvm::json::Object *getPayload() { return payloadVal.getAsObject(); }

  JWT(const JWT &other) = delete;
  JWT(JWT &&other) = default;

private:
  JWT() = default;

  std::string headerStr;
  std::string payloadStr;
  llvm::json::Value headerVal = {};
  llvm::json::Value payloadVal = {};
};
} // namespace

//===----------------------------------------------------------------------===//
// EntitlementStore::generate
//===----------------------------------------------------------------------===//

ErrorOr<EntitlementStore>
EntitlementStore::generate(Config &config, const HTTPContextRef &httpCtx,
                           std::optional<std::string> accessTokenOr,
                           bool openBrowser) {
  std::unique_ptr<HTTPClient> client = httpCtx->client();

  // Key files and certificate paths.
  auto dataFolderOr = M::Config::getModularDataFolderPath(true);
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();
  auto clientKeyPrivPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_key_priv"),
      "client_priv.pem");
  auto clientCertPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_cert"), "client.pem");

  // Generate client keys. The `generate` call here will write the file to
  // disk, and then map it.
  ErrorOr<Keypair> keysOr = Keypair::generate(clientKeyPrivPath);
  if (keysOr.isError())
    return keysOr.takeError();

  std::string accessToken;
  if (accessTokenOr && !accessTokenOr->empty()) {
    accessToken = *accessTokenOr;
  } else {
    // Fetch the token.
    auto toksOr = authAndFetchToken(*client, openBrowser);
    if (toksOr.isError())
      return toksOr.takeError();
    accessToken = std::move(*toksOr);
  }

  auto userInfoOr = requestUserInfo(*client, accessToken);
  if (userInfoOr.isError())
    return userInfoOr.takeError();

  llvm::json::Object *userInfo = userInfoOr->getAsObject();
  if (!userInfo)
    return Error("/v1/oidc/userinfo invalid response, expected JSON object");

  // The user ID is the `sub` field in the token.
  auto commonNameOr = userInfo->getString("sub");
  if (!commonNameOr) {
    return Error("/v1/oidc/userinfo invalid response, expected 'sub'");
  }
  auto tokenIdOr = userInfo->getString("access_token_id");
  StringRef tokenId;
  StringRef commonName = commonNameOr.value();
  if (tokenIdOr) {
    tokenId = tokenIdOr.value();
  } else {
    tokenId = commonNameOr.value();
  }
  Detail::CertSubject subject(commonName.str(), tokenId.str());

  // Fetch the certificate.
  auto certOr = EntitlementStore::fetchCertificate(*client, *keysOr, subject,
                                                   accessToken);
  if (certOr.isError())
    return certOr.takeError();

  // Build our entitlement store; this will reparse the private key.
  auto outOr =
      EntitlementStore::create(keysOr->getBuffer(), std::move(*certOr));
  if (outOr.isError())
    return outOr.takeError();

  // Flush the certificate to a local file now we know it's valid. This is
  // still a fatal error, as `generate` should leave the certificate in place
  // for next time.
  auto writeErr =
      writeFileUnderLock(clientCertPath, [&](llvm::raw_ostream &os) {
        os << outOr->clientCertChain->getPEM();
      });
  if (writeErr.isError())
    return writeErr.takeError();

  return std::move(*outOr);
}

//===----------------------------------------------------------------------===//
// EntitlementStore::alwaysOpen
//===----------------------------------------------------------------------===//

EntitlementStore EntitlementStore::alwaysOpen(llvm::raw_ostream &warnStream) {
  // Open the configuration.
  //
  // N.B. This function is to be removed very shortly in the stack. We can no
  // longer open the entitlement store in this way. The asserts are here
  // temporarily, but will be removed in the future with proper error
  // plumbing.
  auto cfgOr = Config::open();
  assert(!cfgOr.isError());
  auto esOr =
      EntitlementStore::open(*cfgOr, EntitlementStore::getAuthTokenFromEnv());
  if (!esOr.isError() && esOr->has_value())
    return std::move(esOr->value());

  // Return a dummy entitlement store; no entitlements.
  auto null = BufferRef::create(0, std::nullopt, std::nullopt);
  return EntitlementStore(Keypair(), null.copy(), null.copy());
}

//===----------------------------------------------------------------------===//
// EntitlementStore::refresh
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::refresh(Config &config,
                                         const HTTPContextRef &httpCtx) {
  std::unique_ptr<HTTPClient> client = httpCtx->client();

  // Parse the client certificate. It is an error to 'refresh' if we don't
  // already have one.
  if (!clientCertChain || !clientCertChain->isAvailable())
    return Error("no client certificate loaded");

  // Figure out our appropriate key path.
  auto dataFolderOr = M::Config::getModularDataFolderPath(true);
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();
  auto clientKeyPrivPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_key_priv"),
      "client_priv.pem");
  auto clientCertPath = findClientFile(
      *dataFolderOr, config.getValue("entitlements.client_cert"), "client.pem");

  // Step 2 - We do have a client cert, so we should use it to auth to the
  // endpoint to refresh the certificate.

  // Get the subject for the client certificate.
  auto subjectOr = clientCertChain->getSubject();
  if (subjectOr.isError())
    return subjectOr.takeError();

  // Rotate the client's keys on each refresh.
  auto newKeysOr = Keypair::generate(clientKeyPrivPath);
  if (newKeysOr.isError())
    return newKeysOr.takeError();

  // Swap the client keys (the current keys) with the new keys. That way
  // `oldKeys` will point to the old keypair, and `clientKeys` will point to
  // the new keypair for the CSR.
  Keypair oldKeys = std::move(*newKeysOr);
  std::swap(oldKeys, clientKeys);

  // This is the buffer we'll use for the CSR.
  WriteableBufferRef buf = WriteableBuffer::get();
  if (auto err = generateCSR(clientKeys, subjectOr.takeValue(), buf.copy()))
    return err.takeError();

  // Sign the CSR with the old keys and add that signature to the JSON blob.
  auto sigOr = oldKeys.sign(buf->Buffer::getBuffer());
  if (sigOr.isError())
    return sigOr.takeError();
  // Base-64 encode the signature.
  std::string b64Sig = encodeURLSafeBase64(*sigOr);

  // Request the new certificate. This will populate the certificate in
  // memory, but won't verify the certificate.
  auto certOr =
      requestCertificate(*client, buf->Buffer::getBuffer(),
                         /*chain=*/clientCertChain.get(), b64Sig,
                         /*isRefresh=*/true, /*accessToken=*/std::nullopt);
  if (certOr.isError())
    // Include the original error and a helpful message here. It's possible
    // that they haven't actually run anything in a long time, and they are
    // hitting some kind of expiration.
    return Error(Twine("unable to refresh certificate, try `modular auth`: ") +
                 certOr.getError());

  // Update the local certificate.
  auto newCert = std::move(*certOr);
  std::swap(newCert, clientCert);

  // Reparse to update our set of entitlements.
  auto parseErr = parseCertificateChain();
  if (parseErr.isError()) {
    std::swap(newCert, clientCert);
    return parseErr.takeError();
  }

  // Validate and flush the certificate we just got.
  auto writeErr =
      writeFileUnderLock(clientCertPath, [&](llvm::raw_ostream &os) {
        os << clientCertChain->getPEM();
      });
  if (writeErr.isError())
    return writeErr.takeError();

  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::refreshIfNecessary
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::refreshIfNecessary(
    Config &config, const HTTPContextRef &httpCtx,
    llvm::function_ref<bool(std::chrono::system_clock::time_point from,
                            std::chrono::system_clock::time_point to)>
        shouldRefresh) {
  std::unique_ptr<HTTPClient> client = httpCtx->client();

  // It is an error to 'refresh' if we don't already have the client
  // certificate.
  if (!clientCertChain || !clientCertChain->isAvailable())
    return Error("no client certificate loaded");

  // Apply the `shouldRefresh` function to the validity period in the client
  // certificate.
  auto shouldRefreshOr = clientCertChain->applyToValidity(
      shouldRefresh ? shouldRefresh : defaultEntitlementRefreshPolicy);
  if (shouldRefreshOr.isError())
    return shouldRefreshOr.takeError();

  if (*shouldRefreshOr)
    return refresh(config, httpCtx);

  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::parseCertificateChain
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::parseCertificateChain() {
  registerAllEntitlements();

  // Construct a MbedTLSCallbackContext to pass to the extension callback.
  MbedTLSCallbackContext ctx{entitlements, std::nullopt};

  // Parse the certificate, providing the callback to the CertificateChain
  // object to use while it's parsing.
  auto chainOr = M::Detail::CertificateChain::fromPEM(
      (mbedtls_x509_crt_ext_cb_t)extensionCallback, &ctx, clientCert.copy());
  if (chainOr.isError()) {
    if (ctx.error)
      return std::move(*ctx.error);
    return chainOr.takeError();
  }

  // Now that it has been parsed, verify the chain.
  if (auto err = (*chainOr)->verify(clientKeys))
    return err.takeError();

  // Save the certificate chain internally.
  clientCertChain = std::move(*chainOr);
  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::fetchCertificate
//===----------------------------------------------------------------------===//

/// This function uses a token to bootstrap the client certificate.
/// Note that this does not validate the client certificate, since
/// that validation will happen when the cert is read in
/// EntitlementStore::refresh.
ErrorOr<BufferRef>
EntitlementStore::fetchCertificate(HTTPClient &client, Keypair &clientKeys,
                                   const Detail::CertSubject &subject,
                                   std::optional<llvm::StringRef> accessToken) {
  // Generate the CSR. This will be in PEM format.
  WriteableBufferRef csrBuf = WriteableBuffer::get();
  if (auto err = generateCSR(clientKeys, subject, csrBuf.copy()))
    return err.takeError();

  // Great, now we can refresh the cert given the CSR we just generated. Since
  // we aren't rotating the client keypair, we don't want to pass in a
  // signature from a previous key.
  return requestCertificate(client, csrBuf->Buffer::getBuffer(),
                            /*chain=*/nullptr, /*prevKeySig=*/"",
                            /*isRefresh=*/false, /*accessToken=*/accessToken);
}

//===----------------------------------------------------------------------===//
// EntitlementStore::requestCertificate
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> EntitlementStore::requestCertificate(
    HTTPClient &client, StringRef csr,
    M::Detail::CertificateChain *clientCertChain, StringRef prevKeySig,
    bool isRefresh, std::optional<llvm::StringRef> accessToken) {
  auto certURL = isRefresh ? modularAuthURL + "/v1/certificate/renew"
                           : modularAuthURL + "/v1/certificate/issue";
  HTTPRequest certificateRequest{certURL};
  certificateRequest.method = HTTPRequest::Method::POST;
  certificateRequest.headers.try_emplace("content-type", "application/json");
  if (accessToken)
    certificateRequest.accessToken.emplace(*accessToken);

  // Provide the CSR and certificate as PEM-encoded blobs.
  StringRef clientCertChainPEMRef =
      clientCertChain ? clientCertChain->getPEM() : StringRef();
  auto obj = llvm::json::Object({{"certificate_request", csr},
                                 {"certificate", clientCertChainPEMRef},
                                 {"previous_key_signature", prevKeySig}});
  llvm::json::Value val(std::move(obj));

  std::string certRequestBody;
  llvm::raw_string_ostream stream(certRequestBody);
  stream << val;

  certificateRequest.bodyLen = certRequestBody.size();
  certificateRequest.body = ContainerReadCallbackAdaptor(certRequestBody);

  // Perform the request.
  WriteableBufferRef certBuf = WriteableBuffer::get();
  size_t certMaxSize = 4096;
  auto certResponse =
      client.executeRequest(certificateRequest, *certBuf,
                            std::chrono::milliseconds::zero(), certMaxSize);
  if (certResponse.isError())
    return Error(certResponse.asError().getError());

  // Parse the JSON response now.
  auto jsonOr = llvm::json::parse(certBuf->Buffer::getBuffer());
  if (!jsonOr)
    return Error(llvm::toString(jsonOr.takeError()));

  const llvm::json::Object *parsedResponse = jsonOr->getAsObject();
  if (!parsedResponse)
    return Error("expected JSON object in the response");

  auto certOr = parsedResponse->getString("certificate");
  if (!certOr)
    return Error("expected certificate in the response");

  return Buffer::get(*certOr);
}
