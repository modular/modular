# Bazel external repositories provided by nixpkgs instead of being downloaded.
#
# Modular's build declares (bazel/common.MODULE.bazel):
#   @clang-linux-*           prebuilt clang 22 tarballs
#   sysroot-jammy-*          Ubuntu jammy sysroot tarballs
#   @llvm-ifs                prebuilt llvm-ifs
# and rules_python downloads python-build-standalone.  Every one of them is a
# binary download, so they are replaced (--override_repository /
# --override_module in nix/bazelrc.nix) by equivalents built here from
# nixpkgs' source-built packages, preserving Modular's toolchain design
# (unwrapped clang + full sysroot):
#
#   clangRepo      nixpkgs clang 22 / lld / LLVM tools + clang's resource
#                  directory, laid out like the tarball, described by
#                  Modular's own bazel/public-patches/clang.BUILD.
#   sysrootModule  a real sysroot assembled from nixpkgs: glibc (headers,
#                  crt*, libc), gcc (libstdc++ headers+libs, libgcc, crtbegin)
#                  and the system libraries LLDB links (ncurses, libedit,
#                  libbsd) — the nixpkgs analogue of the jammy sysroot.
#   llvmIfsRepo    llvm-ifs / llvm-readtapi from nixpkgs LLVM.
#   pythonRepo     nixpkgs CPython in rules_python's "hermetic interpreter"
#                  layout with the runtime targets rules_python expects.
{
  lib,
  stdenv,
  runCommand,
  llvmPackages_22,
  gcc-unwrapped,
  libgcc,
  glibc,
  buildEnv,
  python3,
  ncurses,
  libedit,
  libbsd,
  libmd,
  zlib,
  src,
}:
let
  llvm = llvmPackages_22;
  clangVersion = lib.versions.major llvm.clang-unwrapped.version; # "22"
  gccVersion = gcc-unwrapped.version;
  clangLib = llvm.clang-unwrapped.lib;
  resourceDir = "${clangLib}/lib/clang/${clangVersion}";
  cpu = stdenv.hostPlatform.parsed.cpu.name; # x86_64 / aarch64
  triple = "${cpu}-unknown-linux-gnu";

  # Runtime library directories baked into every linked binary's rpath
  # (nix/bazelrc.nix): the sysroot paths are execroot-relative at link time,
  # so the runtime search path must point at the store.
  runtimeLibs = [
    "${glibc}/lib"
    "${gcc-unwrapped.lib}/lib"
    "${libgcc}/lib"
    "${ncurses}/lib"
    "${libedit}/lib"
    "${libbsd}/lib"
    "${libmd}/lib"
    "${zlib}/lib"
  ];
  dynamicLinker = "${glibc}/lib/${
    {
      x86_64-linux = "ld-linux-x86-64.so.2";
      aarch64-linux = "ld-linux-aarch64.so.1";
    }
    .${stdenv.hostPlatform.system}
  }";
in
{
  inherit runtimeLibs dynamicLinker;

  clangRepo = runCommand "bazel-repo-clang-${llvm.clang-unwrapped.version}" { } ''
    mkdir -p "$out/bin" "$out/lib/clang/${clangVersion}"
    # Modular's own BUILD file for the tarball (kept in sync automatically).
    cp ${src}/bazel/public-patches/clang.BUILD "$out/BUILD.bazel"

    # Real copy, not a symlink: clang locates ld.lld (-fuse-ld=lld) and its
    # resource directory relative to the *resolved* driver path, which must be
    # this repository's bin/.
    cp ${llvm.clang-unwrapped}/bin/clang-${clangVersion} "$out/bin/clang-${clangVersion}"
    ln -s clang-${clangVersion} "$out/bin/clang"
    ln -s clang-${clangVersion} "$out/bin/clang++"
    ln -s clang-${clangVersion} "$out/bin/clang-cpp"
    ln -s ${llvm.lld}/bin/ld.lld "$out/bin/ld.lld"
    ln -s ${llvm.lld}/bin/lld    "$out/bin/lld"
    for t in ${llvm.llvm}/bin/llvm-* ${llvm.llvm}/bin/dsymutil; do
      [ -e "$t" ] && ln -sf "$t" "$out/bin/$(basename "$t")"
    done
    for t in clang-tidy clang-format clangd clang-scan-deps; do
      [ -e "${llvm.clang-tools}/bin/$t" ] && ln -sf "${llvm.clang-tools}/bin/$t" "$out/bin/$t"
    done

    # Resource directory: builtin headers, sanitizer ignorelists, compiler-rt
    # in both layouts clang may probe.
    cp -rL ${resourceDir}/include "$out/lib/clang/${clangVersion}/include"
    if [ -d ${resourceDir}/share ]; then
      cp -rL ${resourceDir}/share "$out/lib/clang/${clangVersion}/share"
    else
      mkdir "$out/lib/clang/${clangVersion}/share"
    fi
    mkdir -p "$out/lib/clang/${clangVersion}/lib/${triple}" "$out/lib/clang/${clangVersion}/lib/linux"
    for f in ${llvm.compiler-rt}/lib/linux/*; do
      b="$(basename "$f")"
      ln -s "$f" "$out/lib/clang/${clangVersion}/lib/linux/$b"
      ln -s "$f" "$out/lib/clang/${clangVersion}/lib/${triple}/''${b/-${cpu}/}"
    done
    : > "$out/REPO.bazel"
  '';

  # The sysroot: same role as Modular's jammy sysroot tarball, assembled from
  # nixpkgs.  clang finds C headers at usr/include, the GCC installation
  # (crtbegin, libgcc, libstdc++ headers) at usr/lib/gcc/<triple>/<version>
  # and usr/include/c++/<version>, and libraries at usr/lib.
  sysrootModule =
    name:
    runCommand "bazel-module-${name}"
      {
        passAsFile = [ "buildFile" ];
        buildFile = ''
          load("@bazel_skylib//rules/directory:directory.bzl", "directory")

          directory(
              name = "root",
              srcs = glob(["**/*"]),
              visibility = ["//visibility:public"],
          )

          # NOTE: Using this is better for merkle tree performance
          filegroup(
              name = "directory",
              srcs = ["."],
              visibility = ["//visibility:public"],
          )

          filegroup(
              name = "all_files",
              srcs = glob(["**"]),
              visibility = ["//visibility:public"],
          )
        '';
      }
      ''
        mkdir -p "$out/sysroot/usr/include" "$out/sysroot/usr/lib/gcc/${triple}" "$out/sysroot/usr/lib/${triple}"
        cat > "$out/MODULE.bazel" <<EOM
        module(name = "${name}")

        bazel_dep(name = "bazel_skylib", version = "1.7.1")
        EOM
        cp "$buildFilePath" "$out/sysroot/BUILD.bazel"

        # C library headers (glibc's include dir bundles the kernel headers).
        cp -rL ${glibc.dev}/include/. "$out/sysroot/usr/include/"
        # C++ standard library headers.
        mkdir -p "$out/sysroot/usr/include/c++"
        cp -rL ${gcc-unwrapped}/include/c++/${gccVersion} "$out/sysroot/usr/include/c++/${gccVersion}"
        # System libraries LLDB links: ncurses (as curses), libedit, libbsd.
        # (--no-dereference: ncurses' include dir contains self-referential
        # compatibility symlinks like include/ncurses -> '.')
        cp -r --no-dereference ${ncurses.dev}/include/. "$out/sysroot/usr/include/" || true
        cp -rL ${libedit.dev}/include/. "$out/sysroot/usr/include/"
        cp -rL ${libbsd.dev}/include/. "$out/sysroot/usr/include/"
        # Re-materialise symlinked headers as real files, drop self-links.
        find "$out/sysroot/usr/include" -maxdepth 1 -type l | while read -r l; do
          t="$(readlink -f "$l")"
          rm "$l"
          if [ -f "$t" ]; then cp -L "$t" "$l"; fi
        done
        chmod -R u+w "$out/sysroot/usr/include"

        # GCC installation (crt begin/end, libgcc) — what clang's GCC
        # detection expects below usr/lib/gcc/<triple>/<version>.
        mkdir -p "$out/sysroot/usr/lib/gcc/${triple}/${gccVersion}"
        for f in ${gcc-unwrapped}/lib/gcc/${triple}/${gccVersion}/*.o \
                 ${gcc-unwrapped}/lib/gcc/${triple}/${gccVersion}/*.a; do
          ln -s "$f" "$out/sysroot/usr/lib/gcc/${triple}/${gccVersion}/$(basename "$f")"
        done

        # Libraries: glibc (crt1, libc/libm linker scripts, shared objects),
        # libstdc++, libgcc_s, ncurses/libedit/libbsd, zlib.
        for d in ${glibc}/lib ${gcc-unwrapped.lib}/lib ${libgcc}/lib \
                 ${ncurses}/lib ${libedit}/lib ${libbsd}/lib ${libmd}/lib ${zlib}/lib; do
          for f in "$d"/*; do
            [ -f "$f" ] || [ -L "$f" ] && ln -sfn "$(readlink -f "$f")" "$out/sysroot/usr/lib/$(basename "$f")" 2>/dev/null || true
          done
        done
        # glibc's libc.so/libm.so are linker scripts referencing absolute
        # store paths; inside a sysroot, lld insists such paths resolve within
        # the sysroot.  Rewrite them to bare file names, which the linker
        # resolves via its search path (this very directory).
        for script in "$out/sysroot/usr/lib"/*.so; do
          if head -c4 "$script" | grep -qv $'\x7fELF'; then
            tgt="$(readlink -f "$script" 2>/dev/null || echo "$script")"
            if grep -q "GROUP" "$tgt" 2>/dev/null; then
              rm -f "$script"
              sed 's|/nix/store/[^/ )]*/lib/||g' "$tgt" > "$script"
            fi
          fi
        done
        # -lcurses (LLDB) is the wide ncurses in nixpkgs.
        ln -sfn "$(readlink -f ${ncurses}/lib/libncursesw.so)" "$out/sysroot/usr/lib/libcurses.so"
        # lib -> usr/lib, lib64 -> usr/lib: clang/lld search both.
        ln -s usr/lib "$out/sysroot/lib"
        ln -s usr/lib "$out/sysroot/lib64"
      '';

  llvmIfsRepo = runCommand "bazel-repo-llvm-ifs" { } ''
    for p in intel graviton mac; do
      mkdir -p "$out/tools/$p"
      ln -s ${llvm.llvm}/bin/llvm-ifs "$out/tools/$p/llvm-ifs.stripped"
      ln -s ${llvm.llvm}/bin/llvm-readtapi "$out/tools/$p/llvm-readtapi.stripped"
    done
    echo "filegroup(name = 'llvm-ifs', srcs = glob(['**']), visibility = ['//visibility:public'])" > "$out/BUILD.bazel"
    : > "$out/REPO.bazel"
  '';

  pythonRepo =
    let
      py = python3;
      v = py.version; # e.g. 3.14.7
      mm = lib.versions.majorMinor v; # 3.14
      major = lib.versions.major v;
      minor = lib.versions.minor v;
      micro = lib.versions.patch v;
    in
    runCommand "bazel-repo-python-${v}" { } ''
      mkdir -p "$out"
      # The interpreter is used *from this directory* (prefix = repo dir), so
      # bin/, lib/ and include/ are real copies.
      cp -rL ${py}/bin "$out/bin"
      cp -rL ${py}/lib "$out/lib"
      cp -rL ${py}/include "$out/include"
      mkdir -p "$out/share"
      chmod -R u+w "$out"
      rm -rf "$out"/lib/python*/test "$out"/lib/python*/site-packages/__pycache__ 2>/dev/null || true
      ln -sfn bin/python3 "$out/python"
      : > "$out/REPO.bazel"
      echo "# File intentionally left blank. Indicates that this is an interpreter repo created by rules_python." > "$out/STANDALONE_INTERPRETER"

      # Same targets as rules_python's define_hermetic_runtime_toolchain_impl
      # (python/private/hermetic_runtime_repo_setup.bzl), Linux only, plus a
      # stub_shebang that does not need /usr/bin/env.
      cat > "$out/BUILD.bazel" <<BUILD
      # Generated by nix/toolchain-repos.nix — nixpkgs CPython ${v} for rules_python.

      load("@rules_cc//cc:cc_library.bzl", "cc_library")
      load("@rules_python//python:py_runtime.bzl", "py_runtime")
      load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")
      load("@rules_python//python/cc:py_cc_toolchain.bzl", "py_cc_toolchain")
      load("@rules_python//python/private:py_exec_tools_toolchain.bzl", "py_exec_tools_toolchain")

      package(default_visibility = ["//visibility:public"])

      exports_files(["python", "bin/python3"])

      filegroup(
          name = "files",
          srcs = glob(
              include = ["bin/**", "include/**", "share/**", "lib/**"],
              exclude = [
                  "lib/libpython${mm}*.so",
                  "lib/**/*.a",
                  "lib/python${mm}*/**/test/**",
                  "lib/python${mm}*/**/tests/**",
                  "**/__pycache__/*.pyc*",
                  "**/__pycache__/*.pyo*",
              ],
              allow_empty = True,
          ),
      )

      filegroup(
          name = "includes",
          srcs = glob(["include/**/*.h"]),
      )

      cc_library(
          name = "python_headers_abi3",
          hdrs = [":includes"],
          includes = ["include", "include/python${mm}"],
      )

      cc_library(
          name = "python_headers",
          hdrs = [":includes"],
          deps = [":python_headers_abi3"],
      )

      cc_library(
          name = "libpython",
          hdrs = [":includes"],
          srcs = ["lib/libpython${mm}.so", "lib/libpython${mm}.so.1.0"],
      )

      py_runtime(
          name = "py3_runtime",
          files = [":files"],
          interpreter = "bin/python3",
          interpreter_version_info = {
              "major": "${major}",
              "minor": "${minor}",
              "micro": "${micro}",
              "releaselevel": "final",
              "serial": "0",
          },
          python_version = "PY3",
          implementation_name = "cpython",
          pyc_tag = "cpython-${major}${minor}",
          stub_shebang = "#!$out/bin/python3",
      )

      py_runtime_pair(
          name = "python_runtimes",
          py2_runtime = None,
          py3_runtime = ":py3_runtime",
      )

      py_cc_toolchain(
          name = "py_cc_toolchain",
          headers = ":python_headers",
          headers_abi3 = ":python_headers_abi3",
          libs = ":libpython",
          python_version = "${v}",
      )

      py_exec_tools_toolchain(
          name = "py_exec_tools_toolchain",
          precompiler = "@rules_python//tools/precompiler:precompiler",
      )
      BUILD
    '';
}
