# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Validate that we correctly cross compile with a small set of arguments.

# TODO: It should be valid to only pass the --target-triple and get sensible default CPU and features.

# RUN: mojo build --target-triple arm64-apple-macosx11.0 --target-cpu=apple-m1 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_MACOS
# RUN: mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu=x86-64-v3 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_LINUX_X86_64
# RUN: mojo build --target-triple aarch64-unknown-linux-gnu --target-cpu=neoverse-v1 --emit=llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not=ignoring --check-prefix=CHECK_LINUX_AARCH64

# CHECK_MACOS: target triple = "arm64-apple-macosx11.0"
# CHECK_MACOS: "target-cpu"="apple-m1"
# CHECK_MACOS: "target-features"="+aes,+altnzcv,+ccdp,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs"

# CHECK_LINUX_X86_64: target triple = "x86_64-unknown-linux-gnu"
# CHECK_LINUX_X86_64: "target-cpu"="x86-64-v3"
# CHECK_LINUX_X86_64: "target-features"="+avx,+avx2,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fxsr,+lzcnt,+mmx,+movbe,+popcnt,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"

# CHECK_LINUX_AARCH64: target triple = "aarch64-unknown-linux-gnu"
# CHECK_LINUX_AARCH64: "target-cpu"="neoverse-v1"
# CHECK_LINUX_AARCH64: "target-features"="+aes,+bf16,+ccdp,+ccidx,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+sve"


def main():
    pass
