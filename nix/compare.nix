# Report comparing the wheels built from source with the ones Modular
# published for the same version (file lists, sizes, ELF NEEDED entries).
# Not bit-identical by nature (different build machines/toolchain paths), but
# the *shape* of the release must match: same wheels, same files, same
# dynamic dependencies.
{
  lib,
  runCommand,
  python3,
  binutils,
  mojoWheels,
  referenceWheels,
}:
runCommand "modular-release-compare"
  {
    nativeBuildInputs = [
      python3
      binutils
    ];
  }
  ''
    mkdir -p "$out"
    python3 - ${mojoWheels} ${referenceWheels.reference} "$out" <<'PY'
    import os, sys, zipfile, subprocess, re
    ours, theirs, out = sys.argv[1:4]
    def wheels(d):
        # Normalise version AND platform tag: ours are linux_<arch> (Nix-linked
        # binaries), upstream's are manylinux_2_34_<arch>.
        def key(f):
            f = re.sub(r"-[^-]+-py3", "-VERSION-py3", f)
            return re.sub(r"(m?any)?(manylinux[0-9_]+|linux)_(x86_64|aarch64)\.whl$", "PLATFORM.whl", f)
        return {key(f): os.path.join(d, f) for f in os.listdir(d) if f.endswith(".whl")}
    a, b = wheels(ours), wheels(theirs)
    report = []
    ok = True
    for key in sorted(set(a) | set(b)):
        report.append(f"== {key}")
        if key not in a:
            report.append("   MISSING in from-source build"); ok = False; continue
        if key not in b:
            report.append("   (not published upstream)"); continue
        za, zb = zipfile.ZipFile(a[key]), zipfile.ZipFile(b[key])
        norm = lambda n: re.sub(r"-[0-9][^/]*?(\.data|\.dist-info)/", r"-VERSION\1/", n)
        fa = {norm(i.filename): i.file_size for i in za.infolist()}
        fb = {norm(i.filename): i.file_size for i in zb.infolist()}
        for f in sorted(set(fa) | set(fb)):
            if f not in fa:
                report.append(f"   missing: {f}"); ok = f.endswith("gpu-query") and ok
            elif f not in fb:
                report.append(f"   extra:   {f}")
            else:
                ra = fa[f] / fb[f] if fb[f] else 1
                flag = "" if 0.5 < ra < 2 or f.endswith(("RECORD", "METADATA")) else "  <-- size differs a lot"
                report.append(f"   {fa[f]:>12} {fb[f]:>12}  {f}{flag}")
    open(os.path.join(out, "report.txt"), "w").write("\n".join(report) + "\n")
    print("\n".join(report))
    PY
  ''
