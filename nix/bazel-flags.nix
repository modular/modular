# Bazel invocation shared by the dependency-fetch (FOD) and the offline build.
# Both phases MUST evaluate the same configuration so that exactly the same
# set of external repositories is required.
{ lib, versions }:
rec {
  # Options common to `build --nobuild` (fetch) and `build`.
  build = [
    "--curses=no"
    "--color=no"
    "--show_progress_rate_limit=30"
    "--experimental_convenience_symlinks=ignore"

    # Build the compiler from source instead of downloading the nightly.  The
    # flag must also reach the exec configuration (Bazel >= 8 drops Starlark
    # flags there by default), otherwise toolchain resolution for tools would
    # pick the prebuilt nightly wheel -- an unchecksummed download that can
    # neither be cached nor fetched offline.
    "--config=build-mojo"
    "--experimental_propagate_custom_flag=//:use_prebuilt_mojo_toolchain"
    # Release configuration: optimized, assertions off, telemetry pointed at
    # the production endpoint (see //bazel/internal/cc-toolchain/args/modular).
    "--compilation_mode=opt"
    "--//:modular_config=production"
    "--//:release_type=production"
    "--//:modular_version_sha=${versions.rev}"
    "--nostamp"

    # Never talk to Modular's BuildBuddy caches.
    "--remote_cache="
    "--bes_backend="
    "--noremote_upload_local_results"
    "--disk_cache="

    # Bazel 9's repo contents cache stores fully fetched repositories keyed by
    # a hash of their inputs.  We only want the content-addressed download
    # cache (which is what makes the fetch output reproducible), so disable it.
    # (`--repository_cache=` itself is passed by nix/bazel-env.sh.)
    "--repo_contents_cache="

    # Type-checking python with mypy is a lint, not part of the release build.
    "--config=disable-mypy"
  ]
  ++ versions.bazelFlags;

  # Targets making up the from-source Mojo release (see nix/sdk.nix).
  targets = [
    "//KGEN/tools/mojo:mojo-full"
    "//KGEN:CompilerRT"
    "//AsyncRT:RuntimeGlobals"
    "//Support:Globals"
    "@mojo//:std"
    "@llvm-project//lld:lld"
    "@crashpad//:modular-crashpad-handler"
    "//KGEN:MojoLLDB"
    "//KGEN:MojoJupyter"
    "@llvm-project//lldb:lldb24.0.0git"
    "//KGEN:mojo-lldb"
    "//KGEN:gdb-server"
    "@llvm-project//lldb:lldb-dap"
    "@llvm-project//lldb:lldb-argdumper"
    "@llvm-project//llvm:llvm-symbolizer"
    "//KGEN/tools/mojo-lsp-server"
    "//KGEN/tools/mojo-repl-entry-point"
  ];
}
