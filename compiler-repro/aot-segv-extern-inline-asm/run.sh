#!/bin/sh
# Build the C shim and run the reproducer. Exits nonzero if the crash does
# NOT reproduce (so a signal stops the build early). Use --refute to run the
# A/B check that removing `raises` stops the crash.

set -u
cd "$(dirname "$0")"

MOJO=${MOJO:-mojo}
CC=${CC:-cc}

# Both modes link against the shim dylib, so build it up front.
rm -f libshim.dylib shim.o
"$CC" -c shim.c -o shim.o || exit 2
"$CC" -dynamiclib shim.o -o libshim.dylib || exit 2

if [ "$#" -gt 0 ] && [ "$1" = "--refute" ]; then
    # Non-raising control: swap the function to drop `raises` AND remove the
    # raise statements (leaving `raise` in a non-raising fn refuses to
    # compile), so the body is valid. Expect the build to complete (exit 0)
    # instead of SIGSEGV.
    sed -e 's/def do_work(runtime_int: Int) raises:/def do_work(runtime_int: Int):/' \
        -e '/^    if rc != 0:/,/raise Error(/d' \
        repro.mojo > repro_refute.mojo
    rm -f repro_refute
    if "$MOJO" build repro_refute.mojo -I . -Xlinker ./libshim.dylib \
            -o repro_refute 2>/dev/null; then
        echo "REFUTE-OK: non-raising build succeeds (control builds)"
        exit 0
    else
        echo "REFUTE-FAIL: non-raising build also failed"
        exit 1
    fi
fi

rm -f repro
if "$MOJO" build repro.mojo -I . -Xlinker ./libshim.dylib -o repro 2>build.log; then
    echo "CRASH-NOT-REPRODUCED: build succeeded (the bug may be fixed)"
    exit 1
fi
echo "CRASH-REPRODUCED: mojo build did not complete"
exit 0