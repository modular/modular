//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Entitlements/EntitlementStore.h"
#include "Support/Configuration.h"
#include "Support/Cryptography/Keypair.h"
#include "Support/Random.h"
#include "mbedtls/error.h"
#include "mbedtls/x509_crt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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
struct CallbackContext {
  EntitlementStore &store;
  std::optional<Error> error;
};
} // namespace

static constexpr int oidDecodingError = -1;
static constexpr int nonModularOID = -2;
static constexpr int entitlementParsingError = -3;

static int extensionCallback(void *context, mbedtls_x509_crt const *crt,
                             mbedtls_x509_buf *oidBuf, int critical,
                             const unsigned char *dataBegin,
                             const unsigned char *dataEnd) {
  auto *ctx = (CallbackContext *)context;

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

  // Add the entitlement.
  ctx->store.addEntitlement(std::move(*entitlementOr));
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
// getSystemCerts
//===----------------------------------------------------------------------===//

/// Find and parse the system certificates.
static ErrorOrSuccess getSystemRootCerts(mbedtls_x509_crt *list) {
  // Each cert in the array should turn into a cert in the mbedtls CA list - it
  // doesn't treat them as a chain, they are a list.
#ifdef __APPLE__
  // Pull out the trust anchor certificates from the builtin Apple security
  // framework. We have to do a bunch of work bridging CF types and C++ types
  // and stringifying errors.
  CFArrayRef arr;
  OSStatus status = SecTrustCopyAnchorCertificates(&arr);
  if (status != errSecSuccess) {
    CFStringRef errStr = SecCopyErrorMessageString(status, nullptr);
    long errLen = CFStringGetLength(errStr);
    if (errLen <= 0) {
      return Error("unknown error string length, failed to string-ify OSStatus "
                   "error code: " +
                   Twine((int)status));
    }
    // CFStringGetLength returns the number of UTF16 code pairs, not the number
    // of bytes.
    errLen = CFStringGetMaximumSizeForEncoding(errLen, kCFStringEncodingUTF8);
    std::string error;
    error.resize(errLen);
    if (!CFStringGetCString(errStr, error.data(), error.size(),
                            kCFStringEncodingUTF8)) {
      return Error("unable to convert error string to utf8, failed to "
                   "string-ify OSStatus error code: " +
                   Twine((int)status));
    }
    // OK, we were able to string-ify the error code.
    return Error(error);
  }

  auto freeArr = llvm::make_scope_exit([&] { CFRelease(arr); });
  // Now, iterate the CFArrayRef and pull out each cert.
  CFIndex numCerts = CFArrayGetCount(arr);
  if (numCerts <= 0)
    return Error("no trusted root certs found");

  size_t numCertsParsed = 0;
  std::vector<std::string> err;
  for (CFIndex i = 0; i < numCerts; ++i) {
    // Pull the certificate data out of the array.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
    const auto cert = (SecCertificateRef)CFArrayGetValueAtIndex(arr, i);
#pragma clang diagnostic pop
    CFDataRef certDER = SecCertificateCopyData(cert);
    if (certDER == nullptr) {
      err.push_back("could not copy data from root cert " + std::to_string(i));
      continue;
    }

    const uint8_t *certDERBytes = CFDataGetBytePtr(certDER);
    size_t numCertBytes = CFDataGetLength(certDER);

    // Parse the DER into the mbedTLS format.
    int rc = mbedtls_x509_crt_parse_der(list, certDERBytes, numCertBytes);
    // If parsing failed, push the error but otherwise do nothing. If it
    // succeeded, then increment the number of certs we parsed.
    if (rc == 0)
      ++numCertsParsed;
    else
      err.push_back(mbedTLSErrorToString(rc));

    CFRelease(certDER);
  }

  // We didn't parse a single certificate. Return all the errors we must have
  // accumulated.
  if (numCertsParsed == 0) {
    std::string outErr;
    llvm::raw_string_ostream errStream(outErr);
    llvm::interleaveComma(err, errStream);
    return Error(outErr);
  }
  // We parsed at least one cert, so we can say that we "got the system certs".
  return success();
#endif // __APPLE__

#ifdef __linux__
  // Most (all?) linux distros put their ca certificates in a symlink at this
  // path.
  constexpr llvm::StringLiteral linuxCAPath =
      "/etc/ssl/certs/ca-certificates.crt";
  int rc = mbedtls_x509_crt_parse_file(list, linuxCAPath.data());
  if (rc >= 0)
    return success();

  return Error(mbedTLSErrorToString(rc));
#endif // __linux__

#ifdef _WIN32
  HCERTSTORE hStore = CertOpenSystemStore(0, (LPCWSTR)L"ROOT");
  if (hStore == nullptr)
    return Error("could not open system root certificate store");

  std::vector<std::string> errs;
  size_t numCertsParsed = 0;
  PCCERT_CONTEXT pContext = nullptr;
  while ((pContext = CertEnumCertificatesInStore(hStore, pContext)) !=
         nullptr) {
    int rc = mbedtls_x509_crt_parse_der(
        list, (const unsigned char *)pContext->pbCertEncoded,
        pContext->cbCertEncoded);
    if (rc < 0)
      errs.push_back(mbedTLSErrorToString(rc));
    else
      ++numCertsParsed;

    // We don't need to free pContext - CertEnumCertificatesInStore frees it.
  }

  CertCloseStore(hStore, 0);

  if (numCertsParsed == 0) {
    std::string outErr;
    llvm::raw_string_ostream errStream(outErr);
    llvm::interleaveComma(errs, errStream);
    return Error(outErr);
  }

  // We parsed at least one cert, so we can say that we "got the system certs".
  return success();
#endif // _WIN32

  llvm::report_fatal_error(
      "unsupported platform, couldn't parse any root certificates");
}

//===----------------------------------------------------------------------===//
// EntitlementStore
//===----------------------------------------------------------------------===//

/// Provide the platform-specific csprng call.
static int csprng(void *ctx, unsigned char *buf, size_t numBytes) {
  auto *rng = (SecureRandomBytesGenerator *)ctx;
  MutableArrayRef<uint8_t> randBuf(buf, numBytes);
  if (auto err = rng->getRandomBytes(randBuf))
    return 1;
  return 0;
}

ErrorOr<EntitlementStore>
EntitlementStore::open(const std::filesystem::path &clientCertPath,
                       const std::filesystem::path &clientPrivKeyPath,
                       mbedtls_x509_crt *caCerts) {
  auto mbufOr =
      llvm::MemoryBuffer::getFile(clientCertPath.string(), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!mbufOr)
    return Error(mbufOr.getError().message());
  std::unique_ptr<llvm::MemoryBuffer> mbuf = std::move(*mbufOr);

  // Init the certificate chain.
  mbedtls_x509_crt cert;
  mbedtls_x509_crt_init(&cert);
  auto freeCert = llvm::make_scope_exit([&] { mbedtls_x509_crt_free(&cert); });

  EntitlementStore store;

  // Construct a CallbackContext to pass to the extension callback.
  CallbackContext ctx{store, std::nullopt};

  // Parse the client cert into the cert chain. This will also populate the
  // store with the entitlements found in the certificate.
  int rc = mbedtls_x509_crt_parse_der_with_ext_cb(
      &cert, (const unsigned char *)mbuf->getBufferStart(),
      mbuf->getBufferSize(), /*make_copy=*/0,
      /*cb=*/(mbedtls_x509_crt_ext_cb_t)extensionCallback, /*p_ctx=*/&ctx);
  if (rc != 0) {
    if (ctx.error)
      return std::move(*ctx.error);

    return Error(mbedTLSErrorToString(rc));
  }

  // Now the cert is parsed, verify it.

  // TODO(#20699): We have to download the CRLs from the internet...which means
  //               we need to implement caching for it so we don't break in
  //               offline mode.
  mbedtls_x509_crl caCRL;
  mbedtls_x509_crl_init(&caCRL);
  auto freeCRL = llvm::make_scope_exit([&] { mbedtls_x509_crl_free(&caCRL); });

  uint32_t flags = 0;
  // We use the expected next profile mbedTLS provides, but we disallow
  // RSA-2048. This profile targets a 128-bit security level.
  mbedtls_x509_crt_profile profile = mbedtls_x509_crt_profile_next;
  profile.rsa_min_bitlen = 3072;

  // Verify the certificate with the provided security profile. We don't provide
  // additional callbacks, and we don't really care about the common name.
  rc = mbedtls_x509_crt_verify_with_profile(
      &cert, caCerts, &caCRL, &profile,
      /*cn=*/nullptr, &flags, /*f_vrfy=*/nullptr, /*p_vrfy=*/nullptr);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // Now that we know the cert is valid, ensure that we have the private key
  // corresponding to the cert.
  auto privKeyOr = Keypair::openPrivate(clientPrivKeyPath);
  if (privKeyOr.isError())
    return privKeyOr.takeError();
  Keypair privKey = std::move(*privKeyOr);

  // OK - we now have the private key. Ensure the cert was signed by *this*
  // private key. We do this by checking that the public key on the cert matches
  // the private key we just parsed.
  SecureRandomBytesGenerator rng;
  rc = mbedtls_pk_check_pair(&cert.pk, privKey.getRawKey(), &csprng, &rng);
  if (rc != 0)
    return Error(mbedTLSErrorToString(rc));

  // The certificate is valid, so the entitlements we parsed are also valid.
  // Return the entitlement store.
  return store;
}

ErrorOr<EntitlementStore> EntitlementStore::open() {
  // Find the Modular client certificate. This will be encoded as DER.
  auto certOr = findModularFile("client.der");
  if (!certOr)
    return Error("could not find the client certificate");

  auto privOr = findModularFile("client_priv.der");
  if (!privOr)
    return Error("could not find the client private key");

  // Find and open the root certs.
  mbedtls_x509_crt caCerts;
  mbedtls_x509_crt_init(&caCerts);
  auto freeCACerts =
      llvm::make_scope_exit([&] { mbedtls_x509_crt_free(&caCerts); });
  if (auto err = getSystemRootCerts(&caCerts))
    return err.takeError();

  // Then just call the other 'open' method.
  return EntitlementStore::open(*certOr, *privOr, &caCerts);
}

void EntitlementStore::addEntitlement(
    std::unique_ptr<Entitlement> entitlement) {
  entitlements[entitlement->getObjectID()] = std::move(entitlement);
}
