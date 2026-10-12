# Mix generated images into any benchmark workload

This guide describes how to mix generated images into any `benchmark_serving.py`
workload.

## Overview

Every dataset that `benchmark_serving.py` supports (`sonnet`, `sharegpt`,
`instruct-coder`, `arxiv-summarization`, and so on) can now have images mixed
into a fraction of its requests or chat turns. Image mixing runs as a
post-sampling step, so it works the same way regardless of which dataset
you choose.

This is separate from the `random` dataset's own `--random-image-count` and
`--random-image-size` flags, which generate a fixed number and size of images
for every request. Use the general flags described here when you want images
mixed into a fraction of requests, with configurable size and count
distributions, on any dataset. The two mechanisms are mutually exclusive—if
you set both, the benchmark raises an error at startup.

## Quick start

Mix images into 50% of a `sonnet` benchmark's requests:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --backend modular \
  --endpoint /v1/chat/completions \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.5 \
  --image-count 1 \
  --image-long-side 512 \
  --image-aspect-ratio 1.0
```

This selects 50% of requests to carry one generated 512x512 image each. The
remaining requests are unchanged.

## Flag reference

| Flag                   | Description                                                                                                                                                                        | Default |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `--image-fraction`     | Fraction (`0.0`-`1.0`) of requests (single-turn) or chat sessions (multi-turn) that get at least one image.                                                                        | `0.0`   |
| `--image-count`        | Distribution for the number of images on a selected request or turn. Accepts a constant or a distribution string such as `DU(1,4)`.                                                | `1`     |
| `--image-long-side`    | Distribution for each image's longer side, in pixels. Accepts a constant or a distribution string such as `U(224,1024)`.                                                           | `512`   |
| `--image-aspect-ratio` | Distribution for each image's width divided by height. `1.0` produces square images; values above `1.0` produce landscape images; values below `1.0` produce portrait images.      | `1.0`   |
| `--image-turn`         | Which user turn(s) in a multi-turn chat session get images: `first`, `last`, or `every`. Ignored for single-turn requests.                                                         | `first` |
| `--image-num-turns`    | Distribution for the number of user turns in a chat session that gets images. A longer session is cut to the drawn length; a shorter one is kept. Unset keeps sessions as sampled. | unset   |

`--image-fraction` defaults to `0.0`, so none of these flags change behavior
unless you set it above zero.

## Multi-turn sessions

For multi-turn datasets (for example `sharegpt` with `--num-turns` set),
`--image-turn` controls which user turn in each selected session carries the
image:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --backend modular-chat \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --num-prompts 50 \
  --image-fraction 1.0 \
  --image-turn last
```

This attaches an image to the last user turn of every session. Use `every` to
attach an image to every user turn instead of just one.

### What the fraction counts

In a multi-turn run `--image-fraction` selects *sessions*, not requests: exactly
that share of them, rounded at random when it isn't whole, or fewer if too few
sessions can carry images (no user turn, or over the max chat length). That
matters when you
are calibrating against a per-request production figure, because the chat driver
resends a session's history: once a turn carries an image, every later turn of
that session carries it again.

With `--image-turn every` and `--image-count 1` the two coincide — newly encoded
images per request and the share of requests carrying one are equal, and match
`--image-fraction` on average, which is what makes that combination the one to
reach for when matching production. In a single run they're weighted by how many
turns the picked sessions have, so a run that picks long sessions lands above
the fraction. With `first` or `last` one image is encoded per session, so the
encoder rate falls by roughly the average turn count. The share of requests
carrying an image then depends on where that image lands, because only the image
turn and the turns after it resend it:

- `first`: every turn of a selected session carries the image, so the share
  matches `--image-fraction` on average.
- `last`: only the final request of a selected session carries it, so the share
  falls by the same factor as the encoder rate.

The run reports the realized share as `image_request_rate`, in the "Request
Mix" section of its results.

This is deliberately unlike `--response-format-fraction`, which draws per
request and per turn. A structured-output constraint applies only to the
request that sets it, so a per-turn draw lands directly on the share of
requests that set `response_format`. The rule for both: draw per session when
the payload persists into later turns, per turn when it does not.

### Image sessions shorter than the rest

Under `--image-turn every`, an image-bearing request carries more image parts
the longer its session runs, because history is resent. In production,
image-bearing sessions are often shorter than text-only ones.
`--image-num-turns` cuts sessions that get images to a drawn length, which
lowers the parts per image-bearing request. Under `every` and `first` it also
lowers the share of requests carrying an image, so raise `--image-fraction` to
compensate. The run's log reports both, over the turns it sampled.

### `--image-turn first` on a warmed session

`first` is the default, and it is safe on sessions that start mid-conversation
(`--warmup-to-steady-state`, on by default). Those sessions build their opening
turns locally, but the driver splices the images into the history it sends, so
an image on a replayed turn still reaches the server on the first measured
request.

This is where images and `--response-format-turn first` differ: a
`response_format` is a per-request field the prefix loop never sets, so a
constraint on a replayed turn is genuinely lost (CENG-1086). An image is part
of the message history, so it is not.

## Matching a real image-size distribution

`--image-count`, `--image-long-side`, and `--image-aspect-ratio` each accept
any distribution string supported by this codebase's distribution
grammar—not just a constant. Beyond the usual parametric shapes (`N(mean,std)`,
`U(lower,upper)`, `DU(lower,upper)`, `NB(n,p)`, `G(shape,scale)`,
`LN(mean,std)`, `Burr12(c,d,scale)`), there's `Cat(v1:w1, v2:w2, ...)`: an
explicit, weighted set of values, such as image sizes observed in production
traffic.

For example, to draw 70% of images at 1024px, 20% at 512px, and 10% at
2048px on the long side:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.1 \
  --image-long-side "Cat(1024:0.7, 512:0.2, 2048:0.1)" \
  --image-aspect-ratio 1.0
```

The `:weight` suffix is optional per entry—omit it for a uniform choice among
the listed values, so `Cat(1,2,3)` picks each of `1`, `2`, and `3` with equal
probability. Weights don't need to sum to `1`; they're normalized
automatically, so `Cat(1024:7, 512:2, 2048:1)` behaves identically to the
example above.

`Cat(...)` isn't image-specific—it works with any distribution-accepting flag
in this codebase, including `--random-input-len`, `--random-output-len`, and
`--random-num-turns`.

## Checking your workload before running live

Use `--dry-run` to sample the workload and print distribution statistics
without sending any requests to a server. When a workload has images, the
output includes an image-count distribution table, plus a decoded
image-long-side-pixel table when the images are data URIs:

```bash
max benchmark \
  --model google/gemma-3-27b-it \
  --dataset-name sonnet \
  --num-prompts 100 \
  --image-fraction 0.5 \
  --image-count 1 \
  --image-long-side 512 \
  --image-aspect-ratio 1.0 \
  --dry-run
```

```output
================ Workload Statistics ================
  Total requests:                100
  ...

  Image count (per request):
    min       max       mean      std       p5        p25       p50       p75       p95       p99
    0.00      1.00      0.48      0.50      0.00      0.00      0.00      1.00      1.00      1.00

  Image long side (px):
    min       max       mean      std       p5        p25       p50       p75       p95       p99
    512.00    512.00    512.00    0.00      512.00    512.00    512.00    512.00    512.00    512.00
=======================================================
```

The image tables are only printed when the workload actually has images—a run
without `--image-fraction` (or with `--image-fraction 0.0`) shows no image
section. Use this to confirm your flags produce the mix you expect before
spending time against a live server.

> [!NOTE]
> The reported image-count and pixel distributions describe the generated
> workload, not vision-token cost for any specific model. Real token
> accounting for images is architecture-specific (for example, patch-budget
> resizing differs between model families), so treat these tables as a
> workload-shaping tool, not an exact token forecast.
