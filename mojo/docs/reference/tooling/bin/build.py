#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Assemble Mojo cheat-sheet cards from shared chrome + per-card bodies.

Per card, per theme (light + dark), emit:
  mojo-cheat-sheet-<topic>-<light|dark>.pdf   letter PDF (upload / print)
  mojo-cheat-sheet-<topic>-<light|dark>.png   trimmed 2x screenshot (screen)
  mojo-cheat-sheet-<topic>-<light|dark>.svg   vector (docs / devrel / web)

e.g. mojo-cheat-sheet-basics-light.pdf. The content-sized single-page PDF
that seeds the SVG is removed after use. HTML is the source of truth;
everything here is derived from it.

Usage:
    python3 bin/build.py <card>    # one card (light + dark, all formats)
    python3 bin/build.py all       # every card present + combined PDFs

<card> is the <topic> in a src/body-<topic>.html file. The build discovers
cards from the body files present; it never hardcodes a card list.
"""

import html
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # project root; this script lives in bin/
SRC = os.path.join(ROOT, "src")  # hand-edited HTML sources
DIST = os.path.join(ROOT, "dist")  # produced cards (created on demand)


def _find_chrome() -> str:
    """Locate a Chrome or Chromium binary (override with the CHROME_BIN env var)."""
    if os.environ.get("CHROME_BIN"):
        return os.environ["CHROME_BIN"]
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(mac):
        return mac
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return mac  # fall back; a clear error surfaces at render time


CHROME = _find_chrome()
SHEET_W = 1100  # css px; matches .sheet max-width
VERSION = "1.0.0b3"  # bump per release (strip the .devNNN nightly suffix)
PREFIX = "mojo-cheat-sheet"  # filename stem; matches the repo assets dir

# Cards are discovered from the body-<slug>.html files present in src/, so this
# build never hardcodes a card list. Each body file's title and subtitle come
# from two comment lines at its top:
#     <!-- title: ... -->
#     <!-- subtitle: ... -->
META_TITLE = re.compile(r"<!--\s*title:\s*(.*?)\s*-->", re.IGNORECASE)
META_SUB = re.compile(r"<!--\s*subtitle:\s*(.*?)\s*-->", re.IGNORECASE)

KW = set(
    "def struct trait var ref comptime if elif else for while break continue pass return raise try except finally with as from import and or not in is mut out deinit read raises where assert thin abi".split()
)
LIT = set("True False None".split())
TY = set(
    "Int UInt Int8 Int16 Int32 Int64 Int128 Int256 UInt8 UInt16 UInt32 UInt64 UInt128 UInt256 Byte Float16 Float32 Float64 BFloat16 Float8_e4m3fn Float8_e5m2 Float4_e2m1fn Bool String List Dict Optional SIMD Scalar DType Error NoneType StaticString Comparable Copyable Movable Writable Writer ImplicitlyCopyable ImplicitlyDestructible AnyType Equatable Sized Printable PrettyPrintable Identifiable Powable Intable TrivialRegisterPassable RegisterPassable Container Shape Box Pair MyInt Color Point Bag Buffer Matrix Test ValueError Self".split()
)
BI = set("print len range reflect type_of".split())

_Q = chr(34)
_A = chr(39)
_STR = (
    "(?:[rRtT]{1,2})?(?:"
    + _Q * 3
    + r"[\s\S]*?"
    + _Q * 3
    + "|"
    + _A * 3
    + r"[\s\S]*?"
    + _A * 3
    + "|"
    + _Q
    + r"(?:\\.|[^"
    + _Q
    + r"\\\n])*"
    + _Q
    + "|"
    + _A
    + r"(?:\\.|[^"
    + _A
    + r"\\\n])*"
    + _A
    + ")"
)
TOKEN = re.compile(
    r"(#[^\n]*)|(@[A-Za-z_]\w*)|("
    + _STR
    + r")|(\b\d[\w.]*\b)|([A-Za-z_]\w*)|([-+*/%@^&|!<>=~:]+)"
)

LEGEND_DEFS = [
    ("k", "keyword"),
    ("t", "type"),
    ("b", "built-in"),
    ("s", '"string"'),
    ("n", "number"),
    ("o", "operator"),
    ("d", "@decorator"),
    ("l", "True/False/None"),
    ("c", "# comment"),
]


def classes_in(body: str) -> set[str]:
    code = "\n".join(
        re.findall(r'<pre class="code">(.*?)</pre>', body, flags=re.DOTALL)
    )
    code = html.unescape(code)
    found: set[str] = set()
    for m in TOKEN.finditer(code):
        if m.group(1):
            found.add("c")
        elif m.group(2):
            found.add("d")
        elif m.group(3):
            found.add("s")
        elif m.group(4):
            found.add("n")
        elif m.group(5):
            w = m.group(5)
            if w in KW:
                found.add("k")
            elif w in LIT:
                found.add("l")
            elif w in TY:
                found.add("t")
            elif w in BI:
                found.add("b")
        elif m.group(6):
            found.add("o")
    return found


def legend_for(body: str) -> str:
    present = classes_in(body)
    return "\n".join(
        f'    <span><b class="{c}">{lbl}</b></span>'
        for c, lbl in LEGEND_DEFS
        if c in present
    )


def read(p: str, base: str = SRC) -> str:
    with open(os.path.join(base, p)) as f:
        return f.read()


def discover() -> list[str]:
    """Card slugs, from the body-<slug>.html files present in src/."""
    out = []
    for name in sorted(os.listdir(SRC)):
        m = re.match(r"body-(.+)\.html$", name)
        if m:
            out.append(m.group(1))
    return out


def card_meta(slug: str) -> tuple[str, str]:
    """Return (title, subtitle) read from a card body file's top comments."""
    text = read(f"body-{slug}.html")
    t = META_TITLE.search(text)
    s = META_SUB.search(text)
    title = t.group(1) if t else f"Mojo {slug.replace('-', ' ').title()}"
    sub = s.group(1) if s else ""
    return title, sub


def chrome(*flags: str) -> None:
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", *flags],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def build_html(slug: str, dark: bool = False) -> str:
    title, sub = card_meta(slug)
    body = read(f"body-{slug}.html")
    head = (
        read("_head.html")
        .replace("{{TITLE}}", title)
        .replace("{{SUB}}", sub)
        .replace("{{LEGEND}}", legend_for(body))
        .replace("{{VERSION}}", VERSION)
    )
    out = head + "\n" + body + "\n" + read("_foot.html")
    if dark:
        out = out.replace("<body>", '<body class="dark">', 1)
    theme = "dark" if dark else "light"
    stem = f"{PREFIX}-{slug}-{theme}"
    os.makedirs(DIST, exist_ok=True)
    with open(os.path.join(DIST, f"{stem}.html"), "w") as f:
        f.write(out)
    return stem


def render_normal(stem: str, dark: bool) -> None:
    url = f"file://{DIST}/{stem}.html"
    chrome("--no-pdf-header-footer", f"--print-to-pdf={DIST}/{stem}.pdf", url)
    chrome(
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1120,1700",
        f"--screenshot={DIST}/{stem}.png",
        url,
    )
    edge = "#020c13" if dark else "#eceef1"
    subprocess.run(
        [
            "magick",
            f"{DIST}/{stem}.png",
            "-trim",
            "+repage",
            "-bordercolor",
            edge,
            "-border",
            "24",
            f"{DIST}/{stem}.png",
        ],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def measure_sheet_height(stem: str) -> int:
    tmp = f"{DIST}/_measure.png"
    chrome(
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={SHEET_W},4000",
        f"--screenshot={tmp}",
        f"file://{DIST}/{stem}.html",
    )
    subprocess.run(
        ["magick", tmp, "-trim", "+repage", tmp],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    r = subprocess.run(
        ["magick", "identify", "-format", "%h", tmp],
        capture_output=True,
        text=True,
        check=False,
    )
    if os.path.exists(tmp):
        os.remove(tmp)
    h = r.stdout.strip()
    return int(h) if h.isdigit() else 1600


def make_svg(stem: str) -> None:
    h = measure_sheet_height(stem) + 6
    inject = (
        "<style>\n@media print{html,body{font-size:11px;}"
        ".sheet{margin:0;padding:18px 20px 14px;box-shadow:none;max-width:none;}"
        ".cols{column-count:3;column-gap:16px;}.panel{break-inside:avoid;}}\n"
        f"@page{{size:{SHEET_W}px {h}px;margin:0;}}\n</style>\n</head>"
    )
    tmp_html = f"{DIST}/{stem}-1page.html"
    one_pdf = f"{DIST}/{stem}-1page.pdf"
    with open(tmp_html, "w") as f:
        f.write(read(f"{stem}.html", DIST).replace("</head>", inject, 1))
    chrome(
        "--no-pdf-header-footer",
        f"--print-to-pdf={one_pdf}",
        f"file://{tmp_html}",
    )
    svg = f"{DIST}/{stem}.svg"
    subprocess.run(
        ["pdf2svg", one_pdf, svg],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    # pdf2svg outlines glyphs at huge precision (~2.8MB); svgo at precision 1
    # cuts ~70% (~0.9MB) with no visible loss, keeping it under the 2MB repo cap.
    subprocess.run(
        ["npx", "-y", "svgo", "--multipass", "-p", "1", "-i", svg, "-o", svg],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    os.remove(tmp_html)
    os.remove(one_pdf)  # pure SVG source; not a deliverable


def combine(dark: bool = False) -> None:
    theme = "dark" if dark else "light"
    pdfs = [f"{DIST}/{PREFIX}-{slug}-{theme}.pdf" for slug in discover()]
    subprocess.run(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            f"-sOutputFile={DIST}/mojo-cheat-sheets-all-{theme}.pdf",
            *pdfs,
        ],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main() -> None:
    args = sys.argv[1:]
    cards = discover()
    if not args:
        print(__doc__)
        print("cards present:", " ".join(cards) if cards else "(none)")
        return
    ids = cards if args == ["all"] else args
    for slug in ids:
        if not os.path.exists(os.path.join(SRC, f"body-{slug}.html")):
            print("skip unknown card:", slug)
            continue
        for dark in (False, True):
            stem = build_html(slug, dark)
            render_normal(stem, dark)
            make_svg(stem)
            print("built", stem, "(pdf + png + svg)")
    if args == ["all"]:
        combine(False)
        combine(True)
        print(
            "combined letter PDFs -> mojo-cheat-sheets-all-light.pdf + -all-dark.pdf"
        )


if __name__ == "__main__":
    main()
