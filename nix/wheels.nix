# Turns the from-source SDK tree into the same set of wheels that Modular
# publishes for a Mojo release (with a `linux_<arch>` platform tag: the
# binaries are Nix-linked, see below):
#
#   mojo_compiler-<v>-py3-none-<plat>.whl        bin/mojo, bin/lld, crashpad, runtime .so, python `mojo` pkg
#   mojo_compiler_mojo_libs-<v>-py3-none-any.whl  lib/mojo/std.mojoc
#   mojo_lldb_libs-<v>-py3-none-<plat>.whl        libMojoLLDB, libMojoJupyter, liblldb
#   mojo-<v>-py3-none-<plat>.whl                  debugger, LSP, REPL entry point, visualizers
#   mblack-<maxv>-py3-none-any.whl                the Mojo formatter (pure python)
#
# Wheels are built with nix/mkwheel.py, which is fully deterministic (sorted
# entries, fixed timestamps), so `nix build` yields bit-identical files.
{
  lib,
  stdenvNoCC,
  python3,
  sdk,
  src,
  versions,
}:
let
  inherit (stdenvNoCC.hostPlatform) parsed;
  arch = parsed.cpu.name; # x86_64 / aarch64
  # The binaries are ordinary Nix binaries (Nix dynamic linker and rpaths),
  # not manylinux ones: they run wherever the Nix store they were built
  # against exists.  Tag them honestly.
  platformTag = "linux_${arch}";
  mojoV = versions.mojo.full;
  maxV = versions.max.full;

  readme = builtins.readFile (src + "/mojo/README.md");

  metadata =
    {
      name,
      summary,
      requires ? [ ],
      description ? "",
    }:
    ''
      Metadata-Version: 2.1
      Name: ${name}
      Author: Modular Inc
      Author-email: hello@modular.com
      Home-page: https://modular.com
      License: LicenseRef-MAX-Platform-Software-License
      Description-Content-Type: text/markdown
      Summary: ${summary}
      Project-URL: Discord, https://discord.com/invite/modular
      Project-URL: Documentation, https://mojolang.org/docs/
      Project-URL: Forum, https://forum.modular.com/c/mojo
      Project-URL: Issues, https://github.com/modular/modular/issues
      Project-URL: Release notes, https://mojolang.org/releases/
      Project-URL: Source, https://github.com/modular/modular/tree/main/mojo
      ${lib.concatMapStrings (r: "Requires-Dist: ${r}\n") requires}Version: ${
        if name == "mblack" then maxV else mojoV
      }

      ${description}
    '';

  metadataFiles = {
    mojo-compiler = metadata {
      name = "mojo-compiler";
      summary = "The Mojo programming language (compiler only)";
      requires = [ "mojo-compiler-mojo-libs==${mojoV}" ];
      description = readme;
    };
    mojo-compiler-mojo-libs = metadata {
      name = "mojo-compiler-mojo-libs";
      summary = "The Mojo programming language (standard library)";
    };
    mojo-lldb-libs = metadata {
      name = "mojo-lldb-libs";
      summary = "The Mojo programming language (debugger libraries)";
    };
    mojo = metadata {
      name = "mojo";
      summary = "The Mojo programming language";
      requires = [
        "mojo-compiler==${mojoV}"
        "mblack==${maxV}"
        "mojo-lldb-libs==${mojoV}"
      ];
      description = readme;
    };
    mblack = metadata {
      name = "mblack";
      summary = "The Mojo programming language formatter, forked from black";
      requires = [
        "click>=8.0.0"
        "mypy-extensions>=0.4.3"
        "pathspec>=0.9.0"
        "platformdirs>=2"
        "tomli>=1.1.0; python_full_version < \"3.11.0a7\""
      ];
      description = ''
        # mblack

        The [Mojo programming language](https://mojolang.org/) formatter,
        forked from [black](https://pypi.org/project/black/).
      '';
    };
  };

  entryPoints = {
    mojo-compiler = ''
      [console_scripts]
      lld = mojo._entrypoints:exec_lld
      modular-crashpad-handler = mojo._entrypoints:exec_modular_crashpad_handler
      mojo = mojo._entrypoints:exec_mojo
    '';
    mojo = ''
      [console_scripts]
      lldb-argdumper = _mojo._entrypoints:exec_lldb_argdumper
      lldb-dap = _mojo._entrypoints:exec_lldb_dap
      lldb-server = _mojo._entrypoints:exec_lldb_server
      llvm-symbolizer = _mojo._entrypoints:exec_llvm_symbolizer
      mojo-lldb = _mojo._entrypoints:exec_mojo_lldb
      mojo-lsp-server = _mojo._entrypoints:exec_mojo_lsp_server
    '';
    mblack = ''
      [console_scripts]
      mblack = mblack:patched_main
    '';
  };

  writeText = name: text: builtins.toFile name text;
in
stdenvNoCC.mkDerivation {
  pname = "mojo-release-wheels";
  version = mojoV;

  dontUnpack = true;
  nativeBuildInputs = [ python3 ];

  buildPhase = ''
    runHook preBuild
    mk() { python3 ${./mkwheel.py} --out "$out" "$@"; }
    sdk=${sdk}
    root=modular

    # ---- mojo-compiler ----------------------------------------------------
    mkdir -p py/mojo
    cp $sdk/python/mojo/*.py py/mojo/
    mk --name mojo-compiler --version ${mojoV} --tag py3-none-${platformTag} \
      --metadata ${writeText "METADATA" metadataFiles.mojo-compiler} \
      --entry-points ${writeText "entry_points.txt" entryPoints.mojo-compiler} \
      --add py/mojo=mojo \
      --data platlib:$sdk/bin/mojo=$root/bin/mojo \
      --data platlib:$sdk/bin/lld=$root/bin/lld \
      --data platlib:$sdk/bin/modular-crashpad-handler=$root/bin/modular-crashpad-handler \
      --data platlib:$sdk/lib/libAsyncRTRuntimeGlobals.so=$root/lib/libAsyncRTRuntimeGlobals.so \
      --data platlib:$sdk/lib/libKGENCompilerRTShared.so=$root/lib/libKGENCompilerRTShared.so \
      --data platlib:$sdk/lib/libMSupportGlobals.so=$root/lib/libMSupportGlobals.so

    # ---- mojo-compiler-mojo-libs -----------------------------------------
    mk --name mojo-compiler-mojo-libs --version ${mojoV} --tag py3-none-any \
      --metadata ${writeText "METADATA" metadataFiles.mojo-compiler-mojo-libs} \
      --data platlib:$sdk/lib/mojo=$root/lib/mojo

    # ---- mojo-lldb-libs -----------------------------------------------------
    mk --name mojo-lldb-libs --version ${mojoV} --tag py3-none-${platformTag} \
      --metadata ${writeText "METADATA" metadataFiles.mojo-lldb-libs} \
      --data platlib:$sdk/lib/libMojoJupyter.so=$root/lib/libMojoJupyter.so \
      --data platlib:$sdk/lib/libMojoLLDB.so=$root/lib/libMojoLLDB.so \
      --data platlib:$sdk/lib/liblldb24.0.0git.so=$root/lib/liblldb24.0.0git.so

    # ---- mojo -----------------------------------------------------------------
    mkdir -p py/_mojo
    cp ${./python/_mojo/_entrypoints.py} py/_mojo/_entrypoints.py
    mk --name mojo --version ${mojoV} --tag py3-none-${platformTag} \
      --metadata ${writeText "METADATA" metadataFiles.mojo} \
      --entry-points ${writeText "entry_points.txt" entryPoints.mojo} \
      --add py/_mojo=_mojo \
      --data platlib:$sdk/bin/lldb-argdumper=$root/bin/lldb-argdumper \
      --data platlib:$sdk/bin/lldb-dap=$root/bin/lldb-dap \
      --data platlib:$sdk/bin/lldb-server=$root/bin/lldb-server \
      --data platlib:$sdk/bin/llvm-symbolizer=$root/bin/llvm-symbolizer \
      --data platlib:$sdk/bin/mojo-lldb=$root/bin/mojo-lldb \
      --data platlib:$sdk/bin/mojo-lsp-server=$root/bin/mojo-lsp-server \
      --data platlib:$sdk/lib/lldb-visualizers=$root/lib/lldb-visualizers \
      --data platlib:$sdk/lib/mojo-repl-entry-point=$root/lib/mojo-repl-entry-point

    # ---- mblack ----------------------------------------------------------------
    # Same file set as //KGEN/tools/mblack:mblack-lib: src/**/*.py (without
    # __main__.py) plus the src/**/*.txt grammar data.
    mkdir -p py/mblack-src
    cp -r ${src}/KGEN/tools/mblack/src/. py/mblack-src/
    chmod -R u+w py/mblack-src
    rm -f py/mblack-src/mblack/__main__.py
    find py/mblack-src -type f ! -name '*.py' ! -name '*.txt' -delete
    mk --name mblack --version ${maxV} --tag py3-none-any --purelib \
      --metadata ${writeText "METADATA" metadataFiles.mblack} \
      --entry-points ${writeText "entry_points.txt" entryPoints.mblack} \
      --license ${src}/KGEN/tools/mblack/LICENSE \
      --add py/mblack-src=.

    (cd "$out" && sha256sum *.whl > SHA256SUMS)
    runHook postBuild
  '';

  dontInstall = true;

  passthru = { inherit sdk versions platformTag; };

  meta = {
    description = "Mojo release wheels built from source";
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
