# Release version metadata derived from the source tree.
#
# `bazel/mojo.MODULE.bazel` pins the exact Modular release the tree is paired
# with (`MOJO_PACKAGE_VERSION` / `MAX_PACKAGE_VERSION`), i.e. the version of
# the wheels this checkout corresponds to (on a release branch that is the
# release number itself, e.g. `1.0.0` / `26.5.0`; on `main` it is the nightly
# label such as `1.1.0.dev2026081705`).  We reuse it verbatim so that the
# artefacts produced by this flake carry the same names/versions as the ones
# published for the release.
{ lib, src }:
let
  mojoModule = builtins.readFile (src + "/bazel/mojo.MODULE.bazel");
  grab =
    name:
    let
      m = builtins.match ".*${name} = \"([^\"]+)\".*" mojoModule;
    in
    if m == null then throw "versions.nix: cannot find ${name} in bazel/mojo.MODULE.bazel" else builtins.head m;
  split =
    v:
    let
      m = builtins.match "([0-9]+\\.[0-9]+\\.[0-9]+)(.*)" v;
    in
    if m == null then
      throw "versions.nix: cannot parse version ${v}"
    else
      {
        full = v;
        base = builtins.elemAt m 0;
        label = builtins.elemAt m 1;
      };
  mojo = split (grab "MOJO_PACKAGE_VERSION");
  max = split (grab "MAX_PACKAGE_VERSION");
in
{
  inherit mojo max;
  # Short git revision baked into `mojo --version`; falls back to the
  # placeholder Modular uses for unstamped builds.
  rev = if src ? shortRev then src.shortRev else if src ? dirtyShortRev then src.dirtyShortRev else "deadbeef";
  # Bazel build-setting flags carrying the release version into the build.
  bazelFlags = [
    "--//:mojo_base_version=${mojo.base}"
    "--//:mojo_version_label=${mojo.label}"
    "--//:max_base_version=${max.base}"
    "--//:max_version_label=${max.label}"
  ];
}
