# Mach M3 media fixtures

These three H.264 MP4 files are byte-identical copies of the fixtures in
`mach/crates/vision-preprocess/testdata/`. Unlike the similarly named
`mach/crates/multimodal/tests/testdata/m3_clip_headers/` files, these contain
encoded frames, not just container headers. Copies keep the public fuzz
harness independent of the internal Mach build tree.

The `mach_m3_video_stress` scenario samples at 4 FPS (M3 accepts 0.2–5 FPS),
with a 224-pixel long side. Source clips are 8 FPS. Both B-frame and
B-frame-free clips are exercised. A trailing MP4 `free` box varies the
encoded-byte cache key without changing sample offsets or decoded frames.

Select the two accepted-media scenarios explicitly; the profile alone does
not exclude generic scenarios:

```bash
./bazelw run //max/tests/integration/accuracy:llm-fuzz -- \
  --url http://localhost:8091 --model nvidia/MiniMax-M3-NVFP4 \
  --model-profile minimax-m3-mach \
  --scenarios mach_m3_image_stress,mach_m3_video_stress \
  --no-hf-fetch --max-context-length 32768 --max-num-tokens 4096
```

Each scenario sends eight requests at concurrency four, then replays those
bytes to cover warm caches. Requests contain at most 16 items. Fresh bytes
on each repetition cover cold preprocessing. Media rejections are reported
as interesting, never as stress passes; liveness is checked independently.
