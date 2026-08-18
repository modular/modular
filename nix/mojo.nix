# Mojo for Nix: the from-source SDK tree (nix/sdk.nix) with the environment the
# official python entrypoint (`mojo.run._sdk_default_env`) would set — package
# root, import path, driver path — plus a C compiler on PATH because
# `mojo build` uses the system `cc` as its link driver.  The binaries are
# already ordinary Nix binaries (built by nixpkgs' clang/lld), nothing needs
# to be patched.
{
  lib,
  stdenv,
  sdk,
  makeWrapper,
}:
stdenv.mkDerivation {
  pname = "mojo";
  version = sdk.version;
  src = sdk;

  nativeBuildInputs = [ makeWrapper ];

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;
  dontPatchELF = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp -r "$src/bin" "$src/lib" "$out/"
    chmod -R u+w "$out"

    # Keep the real driver at a stable path (it is what tools such as the LSP
    # server and `mojo run` re-exec) and expose a wrapped `bin/mojo`.
    mv "$out/bin/mojo" "$out/bin/.mojo-real"
    makeWrapper "$out/bin/.mojo-real" "$out/bin/mojo" \
      --set-default MODULAR_MOJO_MAX_PACKAGE_ROOT "$out" \
      --set-default MODULAR_MAX_PACKAGE_ROOT "$out" \
      --set-default MODULAR_MOJO_MAX_DRIVER_PATH "$out/bin/.mojo-real" \
      --set-default MODULAR_MOJO_MAX_IMPORT_PATH "$out/lib/mojo" \
      --set-default MODULAR_CRASH_REPORTING_ENABLED 0 \
      --set-default MODULAR_TELEMETRY_ENABLED 0 \
      --prefix PATH : "${lib.makeBinPath [ stdenv.cc ]}"

    for tool in mojo-lldb mojo-lsp-server; do
      mv "$out/bin/$tool" "$out/bin/.$tool-real"
      makeWrapper "$out/bin/.$tool-real" "$out/bin/$tool" \
        --set-default MODULAR_MOJO_MAX_PACKAGE_ROOT "$out" \
        --set-default MODULAR_MAX_PACKAGE_ROOT "$out" \
        --set-default MODULAR_MOJO_MAX_DRIVER_PATH "$out/bin/.mojo-real" \
        --set-default MODULAR_MOJO_MAX_IMPORT_PATH "$out/lib/mojo"
    done
    # Like the wheel's `lldb-dap` entrypoint: pre-load the Mojo LLDB plugin
    # and the LLVM/MLIR data formatters.
    mv "$out/bin/lldb-dap" "$out/bin/.lldb-dap-real"
    makeWrapper "$out/bin/.lldb-dap-real" "$out/bin/lldb-dap" \
      --set-default MODULAR_MOJO_MAX_PACKAGE_ROOT "$out" \
      --set-default MODULAR_MAX_PACKAGE_ROOT "$out" \
      --set-default MODULAR_MOJO_MAX_DRIVER_PATH "$out/bin/.mojo-real" \
      --set-default MODULAR_MOJO_MAX_IMPORT_PATH "$out/lib/mojo" \
      --add-flags "--pre-init-command '?!plugin load $out/lib/libMojoLLDB.so'" \
      --add-flags "--pre-init-command '?command script import $out/lib/lldb-visualizers/lldbDataFormatters.py'" \
      --add-flags "--pre-init-command '?command script import $out/lib/lldb-visualizers/mlirDataFormatters.py'"
    runHook postInstall
  '';

  passthru = {
    inherit sdk;
    unwrapped = sdk;
  };

  meta = {
    description = "The Mojo programming language (compiler, standard library, debugger, LSP) built from source";
    homepage = "https://www.modular.com/mojo";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
    mainProgram = "mojo";
  };
}
