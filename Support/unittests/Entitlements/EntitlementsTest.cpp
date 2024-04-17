//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementStore.h"
#include "Support/Entitlements/EntitlementToken.h"
#include "Support/HTTP/HTTPClient.h"
#include "Support/Random.h"
#include "mbedtls/error.h"
#include "mbedtls/x509_crt.h"
#include "mbedtls/x509_csr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include "RootCert.inc"

using namespace M;

/// This struct provides a simple entitlement.
namespace {
struct TestEntitlement : public Entitlement {
  TestEntitlement() : Entitlement(EK_RESERVED), critical(false) {
    // Simple data, just ensures we can get the same thing we put in.
    for (uint8_t i = 0; i < 32; ++i)
      data.push_back(i);
  }

  TestEntitlement(bool critical, ArrayRef<uint8_t> data)
      : Entitlement(EK_RESERVED), critical(critical), data(data) {
    for (uint8_t i = 0; i < 32; ++i)
      assert(this->data[i] == i);
  }

  static bool classof(const Entitlement *e) {
    return e->getKind() == EK_RESERVED;
  }

  static Kind getKind() { return EK_RESERVED; }

  bool isCritical() const override { return critical; }

  StringRef getName() const override { return "test-entitlement"; }

  static ErrorOr<std::unique_ptr<Entitlement>> create(bool critical,
                                                      ArrayRef<uint8_t> data) {
    return std::make_unique<TestEntitlement>(critical, data);
  }

  std::vector<uint8_t> getDataBytes() override {
    return {data.begin(), data.end()};
  }

  bool critical;
  SmallVector<uint8_t> data;
};
} // namespace

/// Provide the platform-specific csprng call.
static int csprng(void *ctx, unsigned char *buf, size_t numBytes) {
  auto *rng = (SecureRandomBytesGenerator *)ctx;
  MutableArrayRef<uint8_t> randBuf(buf, numBytes);
  if (auto err = rng->getRandomBytes(randBuf))
    return 1;
  return 0;
}

/// mbedTLS writes certificates to the *back* of the buffer, so we have to do
/// some funky pointer gymnastics.
struct WrittenCert {
  // See
  // https://forums.mbed.com/t/does-mbedtls-has-api-to-get-the-size-of-csr-data/5293/7
  // - apparently generally speaking, a 2K buffer is enough.
  std::array<uint8_t, 2048> buf = {};
  size_t bytesWritten = 0;

  ArrayRef<uint8_t> getCertificate() {
    return ArrayRef<uint8_t>(buf.data() + buf.size() - bytesWritten,
                             bytesWritten);
  }
};

class Firechicken : public HTTPClient {
public:
  mbedtls_pk_context keyCtx;
  mbedtls_x509_crt rootCert;

  Firechicken(HTTPContextRef ref) : HTTPClient(std::move(ref)) {
    // For PEM-format objects, the size must include the null terminator (hence
    // the +1).
    mbedtls_pk_init(&keyCtx);
    int rc = mbedtls_pk_parse_key(&keyCtx, modularRootKey.bytes_begin(),
                                  modularRootKey.size() + 1, nullptr, 0,
                                  nullptr, nullptr);
    EXPECT_EQ(rc, 0) << "mbedtls_pk_parse_key failed";

    mbedtls_x509_crt_init(&rootCert);
    rc = mbedtls_x509_crt_parse(&rootCert, modularRootCertificate.bytes_begin(),
                                modularRootCertificate.size() + 1);
    EXPECT_EQ(rc, 0) << "mbedtls_x509_crt_parse failed";
  }
  ~Firechicken() override {
    mbedtls_pk_free(&keyCtx);
    mbedtls_x509_crt_free(&rootCert);
  }

private:
  HTTPResponse executeRequestImpl(
      const HTTPRequest &request, raw_ostream &os,
      std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
      size_t maxLength = 0) override {

    StringRef requestURL(request.URL);

    llvm::unique_function<ErrorOrSuccess(llvm::raw_ostream & os,
                                         const HTTPRequest &request)>
        func;
    if (requestURL.ends_with_insensitive("/oauth/device/authorize")) {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return oauthDeviceAuthorize(os, request);
      };
    } else if (requestURL.ends_with_insensitive("/oauth/token")) {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return oauthToken(os, request);
      };
    } else if (requestURL.ends_with_insensitive("/oidc/userinfo")) {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return userinfo(os, request);
      };
    } else if (requestURL.ends_with_insensitive("/certificate/issue")) {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return issueCertificate(os, request);
      };
    } else if (requestURL.ends_with_insensitive("/certificate/renew")) {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return issueCertificate(os, request);
      };
    } else if (requestURL == "https://crl.modular.com") {
      func = [&](llvm::raw_ostream &os, const HTTPRequest &request) {
        return Error("TODO: implement an actual CRL for this test");
      };
    }

    if (auto err = func(os, request)) {
      return HTTPResponse{
          /*kind=*/HTTPResponse::HTTPResponseError,
          /*responseCode=*/HTTPResponseCode::InternalServerError,
          /*transportErrorMessage=*/err.getError()};
    }

    return HTTPResponse{/*kind=*/HTTPResponse::Success,
                        /*responseCode=*/200,
                        /*transportErrorMessage=*/std::nullopt};
  }

  ErrorOrSuccess oauthDeviceAuthorize(llvm::raw_ostream &os,
                                      const HTTPRequest &request) {
    oauthRegisterCalled = true;

    constexpr llvm::StringLiteral response = R"({
      "device_code": "abcdefg",
      "user_code": "ABCD-EFGH",
      "interval": 5,
      "verification_uri_complete": "https://testing.modular.com"
    })";

    os << response;
    return success();
  }

  ErrorOrSuccess oauthToken(llvm::raw_ostream &os, const HTTPRequest &request) {
    EXPECT_TRUE(oauthRegisterCalled)
        << "the client did not attempt to get the device code";
    // This is a static JWT that contains {"alg": "ES256", "typ": "JWT"}.{"sub":
    // "C=US,O=Modular,CN=mut_abcdefg", "role": "client", "iat":
    // 1516239022}.{sig}. It was generated from the JWT debugger from jwt.io
    // using their default keypair. We use it for both the ID token and the
    // access token.
    constexpr llvm::StringLiteral token =
        "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJtdXRfYWJjZGVmZyIsInJvbGUiOiJjbGllbnQiLCJpYXQiOjE1MTYyMzkwMj"
        "J9.p6XksRx9SLQ_b-Jh1b0lvCnJqXLNIxR9gI8nPhS4mhwceOKgn2_viIQm9t-"
        "JxfKt80v4_ab7LiVJu_nWpRF-IA";
    std::string response = (R"({"id_token": ")" + token +
                            R"(", "access_token": ")" + token + R"("})")
                               .str();

    os << response;
    return success();
  }

  ErrorOrSuccess userinfo(llvm::raw_ostream &os, const HTTPRequest &request) {
    EXPECT_TRUE(oauthRegisterCalled)
        << "the client did not attempt to get the device code";
    constexpr llvm::StringLiteral sub = "mut_abcdefg";
    std::string response = (R"({"sub": ")" + sub + R"("})").str();
    os << response;
    return success();
  }

  ErrorOrSuccess issueCertificate(llvm::raw_ostream &os,
                                  const HTTPRequest &request) {
    // Read the full body. This requires stripping the const because
    std::string bodyData(*request.bodyLen, 0);
    auto bytesWrittenOr = const_cast<HTTPRequest &>(request).body(
        (char *)bodyData.data(), bodyData.size());
    EXPECT_FALSE(bytesWrittenOr.isError()) << bytesWrittenOr.getError();
    if (bytesWrittenOr.isError())
      return bytesWrittenOr.takeError();
    EXPECT_EQ(*bytesWrittenOr, *request.bodyLen);

    auto jsonOr = llvm::json::parse(bodyData);
    EXPECT_TRUE(bool(jsonOr)) << llvm::toString(jsonOr.takeError());

    llvm::json::Object *jsonObj = jsonOr->getAsObject();
    auto csrOr = jsonObj->getString("certificate_request");
    EXPECT_TRUE(csrOr) << "didn't have a CSR";

    auto prevSigOr = jsonObj->getString("previous_key_signature");
    if (jsonObj->getString("certificate")) {
      EXPECT_TRUE(prevSigOr) << "expected to be signed by the previous key if "
                                "we have an old certificate";
      // TODO: Validate the signature using the public key in the certificate.
    }

    mbedtls_x509_csr csr;
    mbedtls_x509_csr_init(&csr);
    auto freeCsr =
        llvm::make_scope_exit([&]() { mbedtls_x509_csr_free(&csr); });

    // PEM encoded stuff needs the +1 in the length since mbedTLS wants us to
    // include the null terminator.
    int rc = mbedtls_x509_csr_parse(&csr, (const uint8_t *)csrOr->str().c_str(),
                                    csrOr->size() + 1);
    EXPECT_EQ(rc, 0) << "mbedtls_x509_csr_parse failed with "
                     << mbedtls_high_level_strerr(rc) << " - "
                     << mbedtls_low_level_strerr(rc);

    mbedtls_x509write_cert cert;
    mbedtls_x509write_crt_init(&cert);
    // Free at the end of the scope.
    auto freeCrt =
        llvm::make_scope_exit([&]() { mbedtls_x509write_crt_free(&cert); });

    // Set up the basic certificate info.
    mbedtls_x509write_crt_set_subject_name(&cert,
                                           "C=US,O=Modular Inc,CN=mut_abcdefg");
    mbedtls_x509write_crt_set_issuer_name(
        &cert, "C=US,O=Modular Inc,CN=dev.auth.modular.com");

    mbedtls_x509write_crt_set_version(&cert, MBEDTLS_X509_CRT_VERSION_3);
    mbedtls_x509write_crt_set_md_alg(&cert, MBEDTLS_MD_SHA256);
    mbedtls_mpi serno;
    mbedtls_mpi_init(&serno);
    auto freeMpi = llvm::make_scope_exit([&]() { mbedtls_mpi_free(&serno); });

    mbedtls_mpi_lset(&serno, 1);
    EXPECT_EQ(mbedtls_x509write_crt_set_serial(&cert, &serno), 0);

    mbedtls_x509write_crt_set_basic_constraints(&cert, /*is_ca=*/0,
                                                /*max_pathlen=*/-1);

    EXPECT_EQ(
        mbedtls_x509write_crt_set_validity(&cert,
                                           /*not_before=*/"20131231235959",
                                           /*not_after=*/"20931231235959"),
        0);

    // Add the entitlement as an extension.
    auto testEntitlement = std::make_unique<TestEntitlement>();
    auto setEntitlement = [&](ArrayRef<uint8_t> oid, bool critical,
                              ArrayRef<uint8_t> data) {
      int rc = mbedtls_x509write_crt_set_extension(
          &cert, (const char *)oid.data(), oid.size(), critical ? 1 : 0,
          data.data(), data.size());
      EXPECT_TRUE(rc == 0) << "mbedtls alloc failed";
    };

    testEntitlement->setAsExtension(setEntitlement);

    // Set the subject and issuer keys.
    mbedtls_x509write_crt_set_subject_key(&cert, &csr.pk);
    mbedtls_x509write_crt_set_issuer_key(&cert, &keyCtx);
    EXPECT_EQ(mbedtls_x509write_crt_set_subject_key_identifier(&cert), 0);
    EXPECT_EQ(mbedtls_x509write_crt_set_authority_key_identifier(&cert), 0);

    mbedtls_x509write_crt_set_key_usage(&cert, csr.key_usage);

    std::array<uint8_t, 2048> outBuf = {};

    // Write the cert to a PEM buffer.
    SecureRandomBytesGenerator rng;
    int bytesWritten = mbedtls_x509write_crt_pem(&cert, outBuf.data(),
                                                 outBuf.size(), &csprng, &rng);
    EXPECT_TRUE(bytesWritten >= 0)
        << "failed to write the PEM of the certificate we just created: "
        << mbedtls_low_level_strerr(bytesWritten);

    // Return the PEM certificate.
    auto obj =
        llvm::json::Object({{"certificate", (const char *)outBuf.data()}});
    llvm::json::Value val(std::move(obj));

    std::string certBody;
    llvm::raw_string_ostream stream(certBody);
    stream << val;

    // Write the response.
    os << certBody;

    return success();
  }

  bool oauthRegisterCalled = false;
};

// HACK: We can only initialize HTTPContext once per process but multiple test
// will need HTTPContext.
static HTTPContextRef getHTTPContextRef() {
  static HTTPContextRef ref;
  static llvm::once_flag flag;
  llvm::call_once(flag, [&]() {
    ref = HTTPContext::init(
        [](HTTPContextRef httpCtx) -> std::unique_ptr<HTTPClient> {
          return std::make_unique<Firechicken>(std::move(httpCtx));
        });
  });
  return ref.copy();
}

TEST(TestEntitlement, Roundtrip) {
  Entitlement::registerEntitlement<TestEntitlement>();

  TestEntitlement test;

  SmallVector<uint8_t> entitlementString;
  auto setEntitlement = [&](ArrayRef<uint8_t> oid, bool critical,
                            ArrayRef<uint8_t> data) {
    entitlementString.push_back(oid.size());
    entitlementString.append(oid.begin(), oid.end());
    entitlementString.push_back(critical);
    entitlementString.append(data.begin(), data.end());
  };

  test.setAsExtension(setEntitlement);

  // The OID is the first [1,buf[0] + 1] bytes.
  ArrayRef<uint8_t> oidbuf(entitlementString.begin() + 1,
                           entitlementString.begin() +
                               *entitlementString.begin() + 1);

  auto oidOr = ASN1::ObjectID::fromEncoded(oidbuf);
  ASSERT_FALSE(oidOr.isError()) << oidOr.getError();
  ASN1::ObjectID oid = std::move(*oidOr);

  // Can't handle non-modular OIDs.
  EXPECT_TRUE(oid.isModularOID()) << "it wasn't a modular OID?";

  // Critical is buf[0] + 2
  bool critical = *(entitlementString.begin() + *entitlementString.begin() + 1);

  // Attempt to parse the entitlement.
  auto entitlementOr =
      Entitlement::parse(oid, critical,
                         ArrayRef<uint8_t>(entitlementString.begin() +
                                               *entitlementString.begin() + 2,
                                           entitlementString.end()));

  ASSERT_FALSE(entitlementOr.isError()) << entitlementOr.getError();
}

/// This test is a bit of a catch-all because many of the pertinent functions
/// are all bundled together in one big test. The EntitlementStore is a useful
/// abstraction over the complexity of dealing with certificates, and we want to
/// avoid forcing it to leak.
TEST(TestEntitlementStore, Bootstrap) {
  Entitlement::registerEntitlement<TestEntitlement>();

  HTTPContextRef httpCtx = getHTTPContextRef();
  Config config; // Use empty config.
  auto storeOr = EntitlementStore::generate(config, httpCtx.copy(), "");
  ASSERT_FALSE(storeOr.isError()) << storeOr.getError();

  auto e = storeOr->getEntitlement<TestEntitlement>();
  EXPECT_TRUE(e != nullptr);
}

/// Bootstrap an EntitlementStore, close it, and open again.
TEST(TestEntitlementStore, BootstrapAndOpen) {
  Entitlement::registerEntitlement<TestEntitlement>();

  HTTPContextRef httpCtx = getHTTPContextRef();
  { // Scope to call the entitlement store's destructor so we can `open` it.
    Config config; // Use empty config.
    auto storeOr = EntitlementStore::generate(config, httpCtx.copy(), "");
    ASSERT_FALSE(storeOr.isError()) << storeOr.getError();
  }

  Config config; // Use empty config.
  auto storeOr = EntitlementStore::open(config);
  ASSERT_FALSE(storeOr.isError()) << storeOr.getError();
  ASSERT_TRUE(storeOr->has_value()) << "we just generated this...?";

  auto e = (*storeOr)->getEntitlement<TestEntitlement>();
  EXPECT_TRUE(e != nullptr);
}

/// Load an EntitlementStore using a token
TEST(TestEntitlementStore, DISABLED_fromToken) {
  Entitlement::registerEntitlement<TestEntitlement>();

  Config config;
  EntitlementToken token;
  // throwaway private key
  token.key =
      "-----BEGIN PRIVATE KEY-----\n"
      "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgIqHReGavAgPFhuoM\n"
      "s6w84NycMM/hFkaWcumQJZa2DCShRANCAAT4V1wTu7QXcPWIHic0P0C25i8QQWUV\n"
      "8fvzSH9oO8BRHEp8tp0DGA+vJ21y/2D0fnVthFjzLKM2uZotZj6tXvQO\n"
      "-----END PRIVATE KEY-----\n";

  // cert signed with "local" certs
  token.certChain.emplace_back(
      "-----BEGIN CERTIFICATE-----\n"
      "MIIBpDCCAUqgAwIBAgIIDa1JmRBfoSkwCgYIKoZIzj0EAwIwQjELMAkGA1UEBhMC\n"
      "VVMxFDASBgNVBAoMC01vZHVsYXIgSW5jMR0wGwYDVQQDDBRkZXYuYXV0aC5tb2R1\n"
      "bGFyLmNvbTAeFw0yNDA0MDExNDU5MzJaFw0yNDA0MDMxNDU5MzJaMDkxEjAQBgNV\n"
      "BAoTCVdheW5lQ29ycDEMMAoGA1UECwwDUiZEMRUwEwYDVQQDDAxtc3RfMTIzNDEy\n"
      "MzQwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAT4V1wTu7QXcPWIHic0P0C25i8Q\n"
      "QWUV8fvzSH9oO8BRHEp8tp0DGA+vJ21y/2D0fnVthFjzLKM2uZotZj6tXvQOozMw\n"
      "MTAOBgNVHQ8BAf8EBAMCB4AwHwYDVR0jBBgwFoAU7y1D961419vhasbaJHQRKGm1\n"
      "QhswCgYIKoZIzj0EAwIDSAAwRQIgDJf+sf0KFwKj7UiDI7WJg9ybAW2ib/w0xhtR\n"
      "J3umlGICIQCdqZCsHtyqL18gNgjOyqVKqgKCd+9YFWmNSqCfK2q1Fw==\n"
      "-----END CERTIFICATE-----\n");

  auto storeOr = EntitlementStore::fromToken(token);
  ASSERT_FALSE(storeOr.isError()) << storeOr.getError();

  auto store = storeOr.takeValue();
  auto key = store.getPrivateKey()->getBuffer().str();
  ASSERT_EQ("-----BEGIN PRIVATE KEY-----\nMIGH", key.substr(0, 32));
}

/// Load an EntitlementStore using a token
TEST(TestEntitlementStore, DISABLED_preferFromToken) {
  Entitlement::registerEntitlement<TestEntitlement>();

  Config config;
  EntitlementToken token;
  // throwaway private key
  token.key =
      "-----BEGIN PRIVATE KEY-----\n"
      "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgIqHReGavAgPFhuoM\n"
      "s6w84NycMM/hFkaWcumQJZa2DCShRANCAAT4V1wTu7QXcPWIHic0P0C25i8QQWUV\n"
      "8fvzSH9oO8BRHEp8tp0DGA+vJ21y/2D0fnVthFjzLKM2uZotZj6tXvQO\n"
      "-----END PRIVATE KEY-----\n";

  // cert signed with "local" certs
  token.certChain.emplace_back(
      "-----BEGIN CERTIFICATE-----\n"
      "MIIBpDCCAUqgAwIBAgIIDa1JmRBfoSkwCgYIKoZIzj0EAwIwQjELMAkGA1UEBhMC\n"
      "VVMxFDASBgNVBAoMC01vZHVsYXIgSW5jMR0wGwYDVQQDDBRkZXYuYXV0aC5tb2R1\n"
      "bGFyLmNvbTAeFw0yNDA0MDExNDU5MzJaFw0yNDA0MDMxNDU5MzJaMDkxEjAQBgNV\n"
      "BAoTCVdheW5lQ29ycDEMMAoGA1UECwwDUiZEMRUwEwYDVQQDDAxtc3RfMTIzNDEy\n"
      "MzQwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAT4V1wTu7QXcPWIHic0P0C25i8Q\n"
      "QWUV8fvzSH9oO8BRHEp8tp0DGA+vJ21y/2D0fnVthFjzLKM2uZotZj6tXvQOozMw\n"
      "MTAOBgNVHQ8BAf8EBAMCB4AwHwYDVR0jBBgwFoAU7y1D961419vhasbaJHQRKGm1\n"
      "QhswCgYIKoZIzj0EAwIDSAAwRQIgDJf+sf0KFwKj7UiDI7WJg9ybAW2ib/w0xhtR\n"
      "J3umlGICIQCdqZCsHtyqL18gNgjOyqVKqgKCd+9YFWmNSqCfK2q1Fw==\n"
      "-----END CERTIFICATE-----\n");

  std::string t = packToken(token);
  auto storeOrOr = EntitlementStore::open(config, t);
  ASSERT_FALSE(storeOrOr.isError()) << storeOrOr.getError();

  auto storeOr = storeOrOr.takeValue();
  ASSERT_TRUE(storeOr.has_value());
  EntitlementStore &store = storeOr.value();
  auto key = store.getPrivateKey()->getBuffer().str();
  // This is the EntitlementToken private key
  ASSERT_EQ("-----BEGIN PRIVATE KEY-----\nMIGH", key.substr(0, 32));
}

/// Check that we can boostrap and then refresh the entitlement certificate.
TEST(TestEntitlementStore, Refresh) {
  Entitlement::registerEntitlement<TestEntitlement>();

  HTTPContextRef httpCtx = getHTTPContextRef();
  Config config; // Use empty config.
  auto storeOr = EntitlementStore::generate(config, httpCtx.copy(), "");
  ASSERT_FALSE(storeOr.isError()) << storeOr.getError();

  auto e = storeOr->getEntitlement<TestEntitlement>();
  EXPECT_TRUE(e != nullptr);

  auto err = storeOr->refresh(config, httpCtx.copy());
  ASSERT_FALSE(err.isError()) << err.getError();

  e = storeOr->getEntitlement<TestEntitlement>();
  EXPECT_TRUE(e != nullptr);
}

/// Currently the refresh policy is to return `true` at the midpoint of `to` and
/// `from`. This test checks this behavior, but will need to be changed if the
/// default policy also changes.
TEST(TestRefreshPolicy, CheckMidpoint) {
  using namespace std::chrono_literals;
  std::chrono::system_clock::time_point from =
      std::chrono::system_clock::now() - 2min;
  std::chrono::system_clock::time_point to3Min =
      std::chrono::system_clock::now() + 3min;
  std::chrono::system_clock::time_point to1Min =
      std::chrono::system_clock::now() + 1min;

  // We are 2 minutes into a 5 minute expiration; no refresh.
  EXPECT_FALSE(defaultEntitlementRefreshPolicy(from, to3Min));

  // We are 2 minutes into a 3 minute expiration; refresh.
  EXPECT_TRUE(defaultEntitlementRefreshPolicy(from, to1Min));
}
