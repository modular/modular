# External artefacts that Nix fetches itself and seeds into Bazel's repository
# cache (see nix/seed.nix).  Everything listed here MUST match the URL/sha256
# used by the corresponding Bazel repository rule, otherwise Bazel would simply
# ignore the seeded entry and try to download it (which fails offline).
#
# Only the big or flaky downloads live here — everything else is fetched by
# Bazel inside the `modular-bazel-deps` fixed-output derivation.
{ lib, fetchurl }:
let
  # LLVM commit and sha256 come from bazel/public-patches/llvm_source.bzl.
  llvmCommit = "ec26997e2e4606d97918a4a082c4f93ca38a6f46";
  llvmSha256 = "7636ff70c60a2933a91362932127478b7b24a610ef1f01afa58fc2a4ecb125ed";
in
[
  rec {
    name = "llvm-project-${llvmCommit}.tar.gz";
    urls = [ "https://github.com/llvm/llvm-project/archive/${llvmCommit}.tar.gz" ];
    sha256 = llvmSha256;
    src = fetchurl {
      inherit name urls sha256;
    };
  }
]
