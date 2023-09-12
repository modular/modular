//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Configuration.h"
#include "Support/Entitlements/Entitlement.h"
#include "Support/Entitlements/EntitlementStore.h"
#include "Support/Random.h"
#include "mbedtls/error.h"
#include "mbedtls/x509_crt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include "Support/Cryptography/Keypair.h"
#include "gtest/gtest.h"

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

  // This is the keypair used for the CA cert.
  Keypair keypair;

  ArrayRef<uint8_t> getCertificate() {
    return ArrayRef<uint8_t>(buf.data() + buf.size() - bytesWritten,
                             bytesWritten);
  }
};

/// Generate a keypair, a certificate, and write both in DER form.
static WrittenCert getCertificate(mbedtls_pk_context *issuer) {
  mbedtls_x509write_cert cert;
  mbedtls_x509write_crt_init(&cert);

  // Set up basic cert info.
  EXPECT_EQ(
      mbedtls_x509write_crt_set_subject_name(&cert, "CN=Cert,O=mbed TLS,C=UK"),
      0);
  EXPECT_EQ(
      mbedtls_x509write_crt_set_issuer_name(&cert, "CN=Cert,O=mbed TLS,C=UK"),
      0);
  mbedtls_x509write_crt_set_version(&cert, MBEDTLS_X509_CRT_VERSION_3);
  mbedtls_x509write_crt_set_md_alg(&cert, MBEDTLS_MD_SHA256);
  mbedtls_mpi serno;
  mbedtls_mpi_init(&serno);
  mbedtls_mpi_lset(&serno, 1);
  EXPECT_EQ(mbedtls_x509write_crt_set_serial(&cert, &serno), 0);
  mbedtls_mpi_free(&serno);

  EXPECT_EQ(mbedtls_x509write_crt_set_validity(&cert,
                                               /*not_before=*/"20131231235959",
                                               /*not_after=*/"20931231235959"),
            0);
  if (!issuer) {
    // This cert is a CA cert. Its max path length is unlimited.
    mbedtls_x509write_crt_set_basic_constraints(&cert, /*is_ca=*/1,
                                                /*max_pathlen=*/-1);
  }

  // Add the entitlement as an extension.
  if (issuer) {
    auto testEntitlement = std::make_unique<TestEntitlement>();
    auto setEntitlement = [&](ArrayRef<uint8_t> oid, bool critical,
                              ArrayRef<uint8_t> data) {
      int rc = mbedtls_x509write_crt_set_extension(
          &cert, (const char *)oid.data(), oid.size(), critical ? 1 : 0,
          data.data(), data.size());
      EXPECT_TRUE(rc == 0) << "mbedtls alloc failed";
    };

    testEntitlement->setAsExtension(setEntitlement);
  }

  WrittenCert out;

  // Generate the keypair and set it on the cert.
  auto keysOr = Keypair::generate();
  EXPECT_FALSE(keysOr.isError()) << keysOr.getError();
  out.keypair = keysOr.takeValue();

  // Set the subject and issuer keys.
  mbedtls_x509write_crt_set_subject_key(&cert, out.keypair.getRawKey());
  mbedtls_x509write_crt_set_issuer_key(&cert, issuer ? issuer
                                                     : out.keypair.getRawKey());
  EXPECT_EQ(mbedtls_x509write_crt_set_subject_key_identifier(&cert), 0);
  EXPECT_EQ(mbedtls_x509write_crt_set_authority_key_identifier(&cert), 0);

  // Write the cert to a DER buffer.
  SecureRandomBytesGenerator rng;
  int bytesWritten = mbedtls_x509write_crt_der(&cert, out.buf.data(),
                                               out.buf.size(), &csprng, &rng);
  EXPECT_TRUE(bytesWritten >= 0)
      << "failed to write the DER of the CSR we just created: "
      << mbedtls_low_level_strerr(bytesWritten);

  mbedtls_x509write_crt_free(&cert);

  out.bytesWritten = bytesWritten;
  return out;
}

TEST(TestEntitlement, Roundtrip) {
  Entitlement::registerEntitlement<TestEntitlement>();

  WrittenCert cert = getCertificate(nullptr);
  ArrayRef<uint8_t> certBuf = cert.getCertificate();

  WrittenCert childCert = getCertificate(cert.keypair.getRawKey());
  ArrayRef<uint8_t> childBuf = childCert.getCertificate();

  // The cert is now self-signed, so we can parse it and parse out the
  // entitlements too.
  mbedtls_x509_crt parsed;
  mbedtls_x509_crt_init(&parsed);
  int rc = mbedtls_x509_crt_parse_der_with_ext_cb(
      &parsed, childBuf.data(), childBuf.size(),
      /*make_copy=*/0,
      [](void *p_ctx, mbedtls_x509_crt const *crt,
         mbedtls_x509_buf const *oidBuf, int critical, const unsigned char *p,
         const unsigned char *end) -> int {
        auto oidOr = ASN1::ObjectID::fromEncoded(
            ArrayRef<uint8_t>(oidBuf->p, oidBuf->len));
        EXPECT_FALSE(oidOr.isError()) << oidOr.getError();
        ASN1::ObjectID oid = std::move(*oidOr);

        // Can't handle non-modular OIDs.
        if (!oid.isModularOID())
          return -1;

        auto entitlementOr =
            Entitlement::parse(oid, bool(critical), ArrayRef<uint8_t>(p, end));
        EXPECT_FALSE(entitlementOr.isError()) << entitlementOr.getError();
        return 0;
      },
      nullptr);
  EXPECT_TRUE(rc == 0) << "failed to parse the certificate DER we just wrote: "
                       << mbedtls_low_level_strerr(rc)
                       << " hl: " << mbedtls_high_level_strerr(rc);

  // Parse the CA cert.
  mbedtls_x509_crt ca;
  mbedtls_x509_crt_init(&ca);
  rc = mbedtls_x509_crt_parse_der_nocopy(&ca, certBuf.data(), certBuf.size());
  EXPECT_TRUE(rc == 0) << "failed to parse the CA cert";

  mbedtls_x509_crl crl;
  mbedtls_x509_crl_init(&crl);

  uint32_t flags = 0;
  rc = mbedtls_x509_crt_verify(&parsed, &ca, &crl, nullptr, &flags, nullptr,
                               nullptr);
  if (rc != 0) {
    std::string errStr(1024, '\0');
    mbedtls_x509_crt_verify_info(errStr.data(), errStr.size(), "", flags);
    ADD_FAILURE() << errStr.c_str();
  }

  mbedtls_x509_crt_free(&parsed);
  mbedtls_x509_crt_free(&ca);
  mbedtls_x509_crl_free(&crl);
}

TEST(TestEntitlementStore, Works) {
  Entitlement::registerEntitlement<TestEntitlement>();

  WrittenCert cert = getCertificate(nullptr);
  ArrayRef<uint8_t> certBuf = cert.getCertificate();

  WrittenCert childCert = getCertificate(cert.keypair.getRawKey());
  ArrayRef<uint8_t> childBuf = childCert.getCertificate();

  // Create a tmp dir under the CWD.
  std::error_code ec;
  std::filesystem::path workdir =
      std::filesystem::absolute("test-entitlement-store-works", ec);
  ASSERT_FALSE(ec) << ec.message();

  std::filesystem::create_directories(workdir, ec);
  ASSERT_FALSE(ec) << ec.message();

  std::string clientCertPath = (workdir / "client.der").string();

  // Write the certificate to the client location so we can read it later.
  auto err = llvm::writeToOutput(clientCertPath, [&](llvm::raw_ostream &os) {
    os.write((const char *)childBuf.data(), childBuf.size());
    return llvm::Error::success();
  });
  EXPECT_FALSE(err) << llvm::toString(std::move(err));

  std::string clientPrivPath = (workdir / "client_priv.der").string();

  // Write the private key in DER form.
  err = llvm::writeToOutput(
      clientPrivPath, [&](llvm::raw_ostream &os) -> llvm::Error {
        std::array<uint8_t, 512> buf = {};
        int bytesWritten = mbedtls_pk_write_key_der(
            childCert.keypair.getRawKey(), buf.data(), buf.size());
        if (bytesWritten <= 0)
          return llvm::createStringError(std::errc::interrupted,
                                         "could not write the keypair to DER");

        os.write((const char *)buf.data() + buf.size() - bytesWritten,
                 bytesWritten);
        return llvm::Error::success();
      });
  EXPECT_FALSE(err) << llvm::toString(std::move(err));

  // The cert is now self-signed, so we can parse it and parse out the
  // entitlements too.
  mbedtls_x509_crt parsed;
  mbedtls_x509_crt_init(&parsed);
  int rc = mbedtls_x509_crt_parse_der_nocopy(&parsed, certBuf.data(),
                                             certBuf.size());
  EXPECT_TRUE(rc == 0) << "failed to parse the certificate DER we just wrote: "
                       << mbedtls_low_level_strerr(rc)
                       << " hl: " << mbedtls_high_level_strerr(rc);

  auto storeOr =
      EntitlementStore::open(clientCertPath, clientPrivPath, &parsed);
  ASSERT_FALSE(storeOr.isError()) << storeOr.getError();

  mbedtls_x509_crt_free(&parsed);

  auto e = storeOr->getEntitlement<TestEntitlement>();
  EXPECT_TRUE(e != nullptr);
}

/// This test checks that we don't open the certificate store without having a
/// valid key by ensuring we return an error on an invalid key.
TEST(TestEntitlementStore, InvalidKey) {
  Entitlement::registerEntitlement<TestEntitlement>();

  WrittenCert cert = getCertificate(nullptr);
  ArrayRef<uint8_t> certBuf = cert.getCertificate();

  WrittenCert childCert = getCertificate(cert.keypair.getRawKey());
  ArrayRef<uint8_t> childBuf = childCert.getCertificate();

  // Create a tmp dir under the CWD.
  std::error_code ec;
  std::filesystem::path workdir =
      std::filesystem::absolute("test-entitlement-store-invalid-key", ec);
  ASSERT_FALSE(ec) << ec.message();

  std::filesystem::create_directories(workdir, ec);
  ASSERT_FALSE(ec) << ec.message();

  std::string clientCertPath = (workdir / "client.der").string();

  // Write the certificate to the client location so we can read it later.
  auto err = llvm::writeToOutput(clientCertPath, [&](llvm::raw_ostream &os) {
    os.write((const char *)childBuf.data(), childBuf.size());
    return llvm::Error::success();
  });
  EXPECT_FALSE(err) << llvm::toString(std::move(err));

  std::string clientPrivPath = (workdir / "client_priv.der").string();

  // Write the incorrect key in DER form.
  auto wrongOr = Keypair::generate(workdir);
  EXPECT_FALSE(wrongOr.isError()) << wrongOr.getError();

  // The cert is now self-signed, so we can parse it and parse out the
  // entitlements too.
  mbedtls_x509_crt parsed;
  mbedtls_x509_crt_init(&parsed);
  int rc = mbedtls_x509_crt_parse_der_nocopy(&parsed, certBuf.data(),
                                             certBuf.size());
  EXPECT_TRUE(rc == 0) << "failed to parse the certificate DER we just wrote: "
                       << mbedtls_low_level_strerr(rc)
                       << " hl: " << mbedtls_high_level_strerr(rc);

  auto storeOr =
      EntitlementStore::open(clientCertPath, clientPrivPath, &parsed);
  EXPECT_TRUE(storeOr.isError());

  mbedtls_x509_crt_free(&parsed);
}
