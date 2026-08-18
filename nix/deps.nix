# Fixed-output derivation holding every external artefact the Bazel build needs:
#
#   $out/repocache          Bazel repository cache (content addressed by sha256;
#                           registry files, module archives, http_archive/http_file
#                           downloads, wheels, ...)
#   $out/MODULE.bazel.lock  the resolved module graph
#
# It is produced by running the very same `bazel build` configuration as the
# real build with `--nobuild`, i.e. loading + analysis, which forces every
# repository (incl. toolchains and aspects) to be fetched.  Because the cache
# only holds checksummed downloads keyed by their sha256, its content — and
# therefore this derivation's output hash — is independent of the machine that
# produced it.  The offline build then consumes it with `--repository_cache`
# and `--lockfile_mode=error`.
#
# Bazel is nixpkgs' (built from source); the compiler, sysroot and CPython
# repositories Modular's build would download are nixpkgs-built as well
# (nix/toolchain-repos.nix, nix/bazelrc.nix), so this cache contains no
# prebuilt binaries — only source archives, module registry files, ...
{
  lib,
  stdenvNoCC,
  cacert,
  bash,
  bazel,
  jdk_headless,
  nixBazelrc,
  src,
  bazelFlags,
  seeds ? [ ],
  hash ? lib.fakeHash,
}:
let
  # Large / flaky downloads are fetched by Nix itself (retriable, mirrorable,
  # substitutable) and seeded into the repository cache before Bazel runs.
  seed = import ./seed.nix { inherit lib; };
in
stdenvNoCC.mkDerivation {
  name = "modular-bazel-deps";
  inherit src;

  outputHashMode = "recursive";
  outputHashAlgo = "sha256";
  outputHash = hash;

  # Downloading may need the proxy settings of the host.
  impureEnvVars = lib.fetchers.proxyImpureEnvVars ++ [ "GIT_PROXY_COMMAND" "SOCKS_SERVER" ];

  nativeBuildInputs = [ bazel ];

  dontUnpack = true;
  dontConfigure = true;
  dontFixup = true;

  SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  NIX_SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";
  NIX_BAZELRC = nixBazelrc;
  NIX_BASH = bash;
  BAZEL_SERVER_JAVABASE = jdk_headless;

  buildPhase = ''
    runHook preBuild
    export repositoryCache="$TMPDIR/repocache"
    mkdir -p "$repositoryCache"
    ${seed.add "$repositoryCache" seeds}

    source ${./bazel-env.sh}
    run_bazel build --nobuild \
      --lockfile_mode=update \
      --loading_phase_threads=4 \
      --experimental_repository_downloader_retries=10 \
      ${lib.escapeShellArgs bazelFlags.build} \
      -- ${lib.escapeShellArgs bazelFlags.targets}
    stop_bazel
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp "$TMPDIR/workspace/MODULE.bazel.lock" "$out/MODULE.bazel.lock"
    cp -r "$repositoryCache" "$out/repocache"
    # The seeded artefacts are inputs of this derivation, not outputs; drop
    # them again so the FOD only carries what Bazel itself downloaded.
    ${seed.remove "$out/repocache" seeds}
    # Bazel 9 may still create the (empty) contents cache directory.
    rm -rf "$out/repocache/contents"
    # No stray temporary files may survive (they would break reproducibility).
    find "$out/repocache" -name '*.tmp' -delete
    chmod -R u+w,a-w "$out"
    runHook postInstall
  '';

  passthru = { inherit seeds bazelFlags; };
}
