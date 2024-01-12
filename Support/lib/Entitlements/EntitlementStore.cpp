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
#include "llvm/Support/raw_ostream.h"

#include "RootCert.inc"

#ifdef __APPLE__
#include <Security/Security.h>
#endif // __APPLE__

#ifdef _WIN32
#include <windows.h>

#include <wincrypt.h>
#endif // _WIN32

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
  ErrorOrSuccess verify(Keypair &clientKeys, StringRef crlPEM);

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
  /// certificate has been verified.
  ErrorOr<std::string> getSubject() const;

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
  auto freePEM = llvm::make_scope_exit([&] { mbedtls_pem_free(&pemCtx); });

  // Parse the chain of certificates. These will be concatenated as PEM
  // buffers one after the other, so we simply parse until we run out of data.
  size_t bytesConsumed = 0;
  for (; bytesConsumed < out->pem->getBufferSize();) {
    // Parse a single certificate first from PEM, then from DER.
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
    if (rc != 0)
      return Error(mbedTLSErrorToString(rc));
  }

  return out;
}

//===----------------------------------------------------------------------===//
// Detail::CertificateChain::verify
//===----------------------------------------------------------------------===//

ErrorOrSuccess Detail::CertificateChain::verify(Keypair &clientKeys,
                                                StringRef crlPEM) {
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
  if (!crlPEM.empty()) {
    int rc =
        mbedtls_x509_crl_parse(&caCRL, crlPEM.bytes_begin(), crlPEM.size());
    if (rc != 0)
      return Error(mbedTLSErrorToString(rc));
  }

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

ErrorOr<std::string> Detail::CertificateChain::getSubject() const {
  assert(verified && "must have a verified certificate chain");
  // The subject is exactly the value of the ASN.1 subject for the previous
  // certificate, which is the user ID. The subject is a linked-list of C/O/CN
  // so we have to find the common name, which is identified by OID 2.5.4.3.
  auto cnOr = ASN1::ObjectID::fromString("2.5.4.3");
  if (cnOr.isError())
    return cnOr.takeError();
  SmallVector<uint8_t> encoded = cnOr->getEncoded();

  auto *commonName = mbedtls_asn1_find_named_data(
      &getLeafCertificate()->subject, (const char *)encoded.data(),
      encoded.size());
  return std::string{(const char *)commonName->val.p, commonName->val.len};
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
  // This request is to an endpoint that requires no auth.
  client.noAuthNeeded();
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
static ErrorOr<std::pair<std::string, std::string>>
pollForOAuthTokens(HTTPClient &client, std::chrono::seconds interval,
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
  std::string accessToken, idToken;
  int maxIters = 256;
  size_t maxSize = 1024;
  // TODO: https://datatracker.ietf.org/doc/html/rfc8628#section-3.5 says we
  //       should unilaterally implement a backoff mechanism if we hit a
  //       timeout.
  while (accessToken.empty() && idToken.empty() && maxIters-- > 0) {
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

    // Pull out the ID token.
    auto idTokOr = responseJSON->getString("id_token");
    if (!idTokOr) {
      return Error("/oauth/token invalid response, expected 'id_token' in "
                   "response payload");
    }
    idToken = *idTokOr;
  }

  // If the token is still empty, return.
  if (accessToken.empty() || idToken.empty()) {
    return Error(
        "max number of requests reached (256) to /oauth/token endpoint");
  }

  return std::make_pair(accessToken, idToken);
}

//===----------------------------------------------------------------------===//
// generateCSR
//===----------------------------------------------------------------------===//

/// Given a keypair and a subject generate a CSR that would generate a
/// certificate of the kind we expect to receive. This generates the data in PEM
/// format because it's always sent directly over the wire. The entitlements are
/// added by the service, so we don't have to fetch the list.
static ErrorOrSuccess generateCSR(Keypair &keys, llvm::StringRef subject,
                                  WriteableBufferRef buf) {
  mbedtls_x509write_csr csr;
  mbedtls_x509write_csr_init(&csr);
  auto freeCSR =
      llvm::make_scope_exit([&] { mbedtls_x509write_csr_free(&csr); });

  // Set the subject name fields.
  auto subjectStr = llvm::formatv("C=US,O=Modular Inc,CN={0}", subject);
  int rc =
      mbedtls_x509write_csr_set_subject_name(&csr, subjectStr.str().c_str());
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

EntitlementStore::EntitlementStore()
    : clientCert(nullptr), crlPEM(Buffer::get("")) {}
EntitlementStore::~EntitlementStore() {}
EntitlementStore::EntitlementStore(M::EntitlementStore &&other) = default;

//===----------------------------------------------------------------------===//
// EntitlementStore::getUserID
//===----------------------------------------------------------------------===//

ErrorOr<std::string>
EntitlementStore::getUserID(std::optional<Config> cfg) const {
  if (clientCert) {
    auto subjOr = clientCert->getSubject();
    if (subjOr.isError())
      return subjOr.takeError();

    return subjOr.takeValue();
  }

  // If we don't have a client cert, then use modular.cfg.
  // TODO(#27787): Phase this out as we roll out entitlements.
  Config config;
  if (!cfg) {
    auto cfgOr = Config::open();
    if (cfgOr.isError())
      return cfgOr.takeError();
    config = cfgOr.takeValue();
  } else {
    config = std::move(*cfg);
  }

  // Get the user ID.
  StringRef val = config.getValue("user.id");
  if (val.empty())
    return Error("no user ID found in modular.cfg");

  return val.str();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::alwaysOpen
//===----------------------------------------------------------------------===//

EntitlementStore EntitlementStore::alwaysOpen(HTTPClient *client,
                                              llvm::raw_ostream &warnStream) {
  if (!client)
    return {}; // Empty entitlement store.

  auto storeOr = EntitlementStore::open(*client);
  if (storeOr.isError()) {
    warnStream << "WARNING: " << storeOr.getError() << "\n";
    return {}; // Empty entitlement store.
  }

  return storeOr.takeValue();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::open
//===----------------------------------------------------------------------===//

ErrorOr<EntitlementStore> EntitlementStore::open(HTTPClient &client) {
  // Register all the entitlements we have.
  registerAllEntitlements();

  EntitlementStore out;
  // Find the client certificate. If we don't have one already, fetch one from
  // auth.modular.com.
  auto certOr = findModularFile("client.pem");
  if (!certOr) {
    if (auto err = out.authAndFetchCertificate(client))
      return err.takeError();
  } else {
    // Otherwise, read the certificate.
    auto mbufOr = Buffer::getFile(certOr->string());
    if (mbufOr.isError())
      return mbufOr.takeError();

    // Parse the certificate chain. This will store the cert in this class.
    if (auto err = out.parseCertificateChain(std::move(*mbufOr)))
      return err.takeError();
  }

  // Open the keypair, this function prefers a private key.
  auto privKeyOr = Keypair::open();
  if (privKeyOr.isError())
    return privKeyOr.takeError();
  if (!privKeyOr->hasPrivateKey())
    return Error("client keypair did not include a private key");
  out.clientKeys = std::move(*privKeyOr);

  // Validate the certificate.
  if (auto err = out.verifyAndFlushClientCert(client))
    return err.takeError();

  return out;
}

//===----------------------------------------------------------------------===//
// EntitlementStore::openWithRetry
//===----------------------------------------------------------------------===//

ErrorOr<EntitlementStore> EntitlementStore::openWithRetry(HTTPClient &client) {
  // Attempt to open, if it worked then return it.
  auto storeOr = open(client);
  if (!storeOr.isError())
    return std::move(storeOr);

  // Otherwise, remove the client cert file (if it exists) and try again.
  auto certFile = findModularFile("client.pem");
  if (certFile) {
    std::error_code ec;
    std::filesystem::remove(*certFile, ec);
  }

  return open(client);
}

//===----------------------------------------------------------------------===//
// EntitlementStore::refresh
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::refresh(HTTPClient &client) {
  // Parse the client certificate. It is an error to 'refresh' if we don't
  // already have one.
  if (!clientCert->isAvailable())
    return Error("no client certificate loaded");

  // Step 2 - We do have a client cert, so we should use it to auth to the
  // endpoint to refresh the certificate.

  // Get the subject for the client certificate.
  auto subjectOr = clientCert->getSubject();
  if (subjectOr.isError())
    return subjectOr.takeError();

  // Set up auth - this will read the certificate from the filesystem and use
  // that. This should still use the old keys for now.
  if (auto err = client.setupAuth())
    return err.takeError();

  // Ensure we have a data folder.
  auto dataFolderOr = Config::getModularDataFolderPath();
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();

  // Rotate the client's keys on each refresh.
  auto newKeysOr = Keypair::generate(*dataFolderOr);
  if (newKeysOr.isError())
    return newKeysOr.takeError();

  // Swap the client keys (the current keys) with the new keys. That way
  // `oldKeys` will point to the old keypair, and `clientKeys` will point to the
  // new keypair for the CSR.
  Keypair oldKeys = std::move(*newKeysOr);
  std::swap(oldKeys, clientKeys);

  // This is the buffer we'll use for the CSR.
  WriteableBufferRef buf = WriteableBuffer::get();
  if (auto err = generateCSR(clientKeys, *subjectOr, buf.copy()))
    return err.takeError();

  // Sign the CSR with the old keys and add that signature to the JSON blob.
  auto sigOr = oldKeys.sign(buf->Buffer::getBuffer());
  if (sigOr.isError())
    return sigOr.takeError();
  // Base-64 encode the signature.
  std::string b64Sig = encodeURLSafeBase64(*sigOr);

  // Request the new certificate. This will populate the certificate in memory,
  // but won't verify the certificate.
  if (auto err = requestCertificate(client, buf->Buffer::getBuffer(), b64Sig,
                                    /*isRefresh=*/true))
    return err.takeError();

  // Validate and flush the certificate we just got.
  if (auto err = verifyAndFlushClientCert(client))
    return err.takeError();

  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::refreshIfNecessary
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::refreshIfNecessary(
    HTTPClient &client,
    llvm::function_ref<bool(std::chrono::system_clock::time_point from,
                            std::chrono::system_clock::time_point to)>
        shouldRefresh) {
  // It is an error to 'refresh' if we don't already have the client
  // certificate.
  if (!clientCert->isAvailable())
    return Error("no client certificate loaded");

  // Apply the `shouldRefresh` function to the validity period in the client
  // certificate.
  auto shouldRefreshOr = clientCert->applyToValidity(shouldRefresh);
  if (shouldRefreshOr.isError())
    return shouldRefreshOr.takeError();

  if (*shouldRefreshOr)
    return refresh(client);

  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::verifyAndFlushClientCert
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::verifyAndFlushClientCert(HTTPClient &client) {
  // Ensure that we have a data folder.
  auto dataFolderOr = Config::getModularDataFolderPath();
  if (dataFolderOr.isError())
    return dataFolderOr.takeError();

  HTTPRequest certificateRequest{"https://crl.modular.com"};
  WriteableBufferRef crlBuf = WriteableBuffer::get(
      /*size=*/0, /*alignment=*/std::nullopt, /*capacity=*/2048);
  // The CRL isn't behind any kind of auth - that's allowed to be public.
  client.noAuthNeeded();
  HTTPResponse response = client.executeRequest(certificateRequest, *crlBuf);
  // We ignore failures here, for now at least, it isn't an error to fail to
  // fetch the CRL.
  if (response.isSuccess()) {
    // Flush it to the filesystem.
    auto err = writeFileUnderLock(*dataFolderOr / "crl.pem",
                                  [&](llvm::raw_ostream &os) {
                                    os << crlBuf->Buffer::getBuffer();
                                    // Write a null-terminator explicitly - this
                                    // is required by mbedTLS' PEM parsing
                                    // functions.
                                    os << '\0';
                                  });
    if (err.isError()) {
      return err.takeError();
    }
  }

  // If we got a new CRL, we'll have it on the filesystem. If we didn't, but we
  // had an old one, then we'll be able to pull it off the filesystem. If we
  // have nothing, then we can simply parse nothing.
  auto crlPEMPathOr = findModularFile("crl.pem");
  if (crlPEMPathOr) {
    auto pemOr = Buffer::getFile(*crlPEMPathOr);
    if (!pemOr.isError())
      crlPEM = std::move(*pemOr);
  }

  // Verify the certificate with the CRL we fetched.
  if (auto err = clientCert->verify(clientKeys, crlPEM->getBuffer()))
    return err.takeError();

  // Flush the certificate to a local file now we know it's valid.
  auto err = writeFileUnderLock(
      *dataFolderOr / "client.pem",
      [&](llvm::raw_ostream &os) { os << clientCert->getPEM(); });
  if (err.isError())
    return err.takeError();

  // The certificate is valid, so the entitlements we parsed are also valid.
  // Return the entitlement store.
  return success();
}

//===----------------------------------------------------------------------===//
// EntitlementStore::parseCertificateChain
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::parseCertificateChain(BufferRef buf) {
  // Construct a MbedTLSCallbackContext to pass to the extension callback.
  MbedTLSCallbackContext ctx{entitlements, std::nullopt};

  // Parse the certificate, providing the callback to the CertificateChain
  // object to use while it's parsing.
  auto chainOr = Detail::CertificateChain::fromPEM(
      (mbedtls_x509_crt_ext_cb_t)extensionCallback, &ctx, std::move(buf));
  if (chainOr.isError()) {
    if (ctx.error)
      return std::move(*ctx.error);

    return chainOr.takeError();
  }

  // All done, move it to this class (for storage) and return.
  clientCert = std::move(*chainOr);
  return success();
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
// EntitlementStore::authAndFetchCertificate
//===----------------------------------------------------------------------===//

/// This function uses the OAuth Device Authorization Flow to do initial
/// authentication and bootstrap the client certificate. Note that this does not
/// validate the client certificate, since that validation will happen when the
/// cert is read in EntitlementStore::refresh.
ErrorOrSuccess EntitlementStore::authAndFetchCertificate(HTTPClient &client) {
  auto jsonOr = requestDeviceCode(client);
  if (jsonOr.isError())
    return jsonOr.takeError();

  llvm::json::Object *jsonResponse = jsonOr->getAsObject();
  if (!jsonResponse)
    return Error("/oauth/device/code invalid response, expected JSON object");

  auto codeOr = jsonResponse->getString("device_code");
  if (!codeOr) {
    return Error(
        "/oauth/device/code invalid response, expected 'device_code' in "
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

  llvm::outs() << "# Please visit this URL in your browser: " << *verifURIOr
               << "\n";
  llvm::outs() << "# Waiting for confirmation...\n";

  auto toksOr =
      pollForOAuthTokens(client, std::chrono::seconds(*intervalOr), *codeOr);
  if (toksOr.isError())
    return toksOr.takeError();
  auto [accessToken, idToken] = std::move(*toksOr);

  // Set up the auth provider with the access token. This will set up auth for
  // all further usage of this client.
  if (auto err = client.setupAuth(accessToken))
    return err.takeError();

  // Now, parse the ID token to get the user ID.
  auto jwtOr = JWT::parse(idToken);
  if (jwtOr.isError())
    return jwtOr.takeError();
  JWT jwt = std::move(*jwtOr);

  // The user ID is the `sub` field in the token.
  auto subjectOr = jwt.getPayload()->getString("sub");
  if (!subjectOr) {
    return Error("/oauth/token invalid response, expected 'sub' in the "
                 "returned ID token");
  }

  // OK, we have a token now. This means we can auth to the CSR endpoint.
  // Open or generate some keys and use them to generate our CSR.
  auto keysOr = Keypair::open();

  // Couldn't find the default keys, so create new ones and write them to the
  // MODULAR_HOME path. This will ensure we have the keys on the filesystem at
  // all times.
  if (keysOr.isError()) {
    auto configFolderOr = Config::getModularConfigFolderPath();
    if (configFolderOr.isError())
      return configFolderOr.takeError();
    keysOr = Keypair::generate(*configFolderOr);
  }

  // Now if it's an error, it's a real error.
  if (keysOr.isError())
    return keysOr.takeError();

  // Now we have our keys.
  clientKeys = std::move(*keysOr);

  // Generate the CSR. This will be in PEM format.
  WriteableBufferRef csrBuf = WriteableBuffer::get();
  if (auto err = generateCSR(clientKeys, *subjectOr, csrBuf.copy()))
    return err.takeError();

  // Great, now we can refresh the cert given the CSR we just generated. Since
  // we aren't rotating the client keypair, we don't want to pass in a signature
  // from a previous key.
  return requestCertificate(client, csrBuf->Buffer::getBuffer(),
                            /*prevKeySig=*/"", /*isRefresh=*/false);
}

//===----------------------------------------------------------------------===//
// EntitlementStore::requestCertificate
//===----------------------------------------------------------------------===//

ErrorOrSuccess EntitlementStore::requestCertificate(HTTPClient &client,
                                                    StringRef csr,
                                                    StringRef prevKeySig,
                                                    bool isRefresh) {
  auto certURL = isRefresh ? modularAuthURL + "/v1/certificate/renew"
                           : modularAuthURL + "/v1/certificate/issue";
  HTTPRequest certificateRequest{certURL};
  certificateRequest.method = HTTPRequest::Method::POST;
  certificateRequest.headers.try_emplace("content-type", "application/json");

  // Provide the CSR and certificate as PEM-encoded blobs.
  StringRef clientCertChainPEMRef =
      clientCert ? clientCert->getPEM() : StringRef();
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
    return certResponse.asError();

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

  // Parse the certificate chain we just received.
  if (auto err = parseCertificateChain(Buffer::get(*certOr)))
    return err.takeError();

  // Success! We have the certificate chain now, so we're done.
  return success();
}
