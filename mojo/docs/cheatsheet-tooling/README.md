# Mojo cheat sheets — build kit

This folder builds the Mojo language cheat sheets: one-page reference cards
rendered from HTML sources to PDF, PNG, and SVG in light and dark themes. Edit a
card's source, run the build, and its files regenerate.

The published downloads live in `reference/assets/`. The page that links them is
`reference/cheat-sheets.mdx`.

## Layout

```text
tooling/
  README.md            this file
  SKILL.md             Claude Code workflow for updating a card
  bin/build.py         assembles + renders cards
  src/                 hand-edited sources
    _head.html         shared CSS, palette, header
    _foot.html         shared close
    body-<topic>.html  one file per card
  dist/                produced files (generated; not committed)
```

The build discovers cards from the `body-<topic>.html` files present, so there
is no card list to maintain.

## Required tools

| tool          | used for                       | install                                 |
|---------------|--------------------------------|-----------------------------------------|
| Python 3      | runs `build.py`                | system                                  |
| Google Chrome | headless render to PDF + PNG   | google.com/chrome (or set `CHROME_BIN`) |
| ImageMagick   | trim + measure PNGs (`magick`) | `brew install imagemagick`              |
| pdf2svg       | PDF to SVG                     | `brew install pdf2svg`                  |
| ghostscript   | combine per-card PDFs (`gs`)   | `brew install ghostscript`              |
| Node / npx    | runs `svgo` to shrink SVGs     | `brew install node`                     |

Fonts: the cards use **Inter** (text) and **Roboto Mono** (code). Install both
locally; Chrome renders them at build time. Fonts are not part of this kit.

## Build

```bash
python3 bin/build.py <topic>   # one card (light + dark, all formats)
python3 bin/build.py all       # every card present + combined PDFs
```

`<topic>` is the name in `src/body-<topic>.html`. Run `python3 bin/build.py`
with no arguments to list the cards currently present. Output filenames follow
`mojo-cheat-sheet-<topic>-<light|dark>.<ext>`.

## Card shape

Each card chooses a layout preset with an optional comment at the top of its
body file, defaulting to portrait:

```text
<!-- layout: portrait -->   2 columns, ~900px wide, portrait PDF (the default)
<!-- layout: landscape -->  3 columns, ~1100px wide, landscape PDF
```

Override a single knob when a card needs more room:

```text
<!-- columns: 4 -->
<!-- width: 1300 -->
```

Content-light cards read best portrait; dense cards (many panels, wide tables)
read best landscape. A card grows taller on its own as panels are added.

## Update a card

1. Edit `src/body-<topic>.html`. Never edit files in `dist/`.
2. Keep each card's title, subtitle, and optional `layout` in the comment lines
   at the top of its body file (see Card shape).
3. Verify every behavioral claim against the Mojo reference docs
   (`mojo/docs/reference/`) or a runnable compiler check.
4. Rebuild that card and open the PNG to review it.

See `SKILL.md` for the full workflow.

## Publishing

Regenerating the published downloads and updating the site is a separate,
deliberate step handled by the maintainers, not part of normal editing. Commit
your source changes and open a pull request.
