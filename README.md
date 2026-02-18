<div align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/modular_github_logo_bg.png">

[About Modular] | [Get started] | [API docs] | [Contributing]
  | [Changelog] | [MAX Model Development]
</div>

[About Modular]: https://www.modular.com/
[Get started]: https://docs.modular.com/max/get-started
[API docs]: https://docs.modular.com/max/api
[Contributing]: ./CONTRIBUTING.md
[Changelog]: https://docs.modular.com/max/changelog
[MAX Model Development]: /max/docs/development.md

---
🔥 [Modular has acquired BentoMl][bentoml-joins] to extend our open source AI stack!
[Join us][bentoml-joins-ama] on February 17th at
[9:30-11:30am PT][ama-time] for an Ask Us Anything with Chris Lattner and
BentoML Founder Chaoyu Yang [in the Modular forum][bentoml-joins-ama].
We’ll answer questions and share more about our plans!
[Feel free to share your questions now.][bentoml-joins-ama]

# Modular Platform

> A unified platform for AI development and deployment, including **MAX**🧑‍🚀 and
**Mojo**🔥.

The Modular Platform is an open and fully-integrated suite of AI libraries
and tools that accelerates model serving and scales GenAI deployments. It
abstracts away hardware complexity so you can run the most popular open
models with industry-leading GPU and CPU performance without any code changes.

![](https://docs.modular.com/images/modular-container-stack.png?20250513)

## Get started

You don't need to clone this repo.

You can install Modular as a `pip` or `conda` package and then start an
OpenAI-compatible endpoint with a model of your choice.

To get started with the Modular Platform and serve a model using the MAX
framework, see [the quickstart guide](https://docs.modular.com/max/get-started).

> [!NOTE]
> **Nightly vs. stable releases**
> If you cloned the repo and want a stable release, run
  `git checkout modular/vX.X` to match the version.
> The `main` branch tracks nightly builds, while the `stable` branch matches
  the latest released version.

After your model endpoint is up and running, you can start sending the model
inference requests using
[our OpenAI-compatible REST API](https://docs.modular.com/max/api/serve).

Try running hundreds of other models from
[our model repository](https://builds.modular.com/?category=models).

## Deploy our container

The MAX container is our Kubernetes-compatible Docker container for convenient
deployment, which uses the MAX framework's built-in inference server. We have
separate containers for NVIDIA and AMD GPU environments, and a unified container
that works with both.

For example, you can start a container for an NVIDIA GPU with this command:

```sh
docker run --gpus=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    modular/max-nvidia-full:latest \
    --model-path google/gemma-3-27b-it
```

For more information, see our [MAX container
docs](https://docs.modular.com/max/container) or the [Modular Docker Hub
repository](https://hub.docker.com/u/modular).

## About the repo

We're constantly open-sourcing more of the Modular Platform and you can find
all of it in here. As of May, 2025, this repo includes over 450,000 lines of
code from over 6000 contributors, providing developers with production-grade
reference implementations and tools to extend the Modular Platform with new
algorithms, operations, and hardware targets.

It's quite likely **the world's largest repository of open source CPU and GPU
kernels**!

Highlights include:

- Mojo standard library: [/mojo/stdlib](mojo/stdlib)
- MAX GPU and CPU kernels: [/max/kernels](max/kernels) (Mojo kernels)
- MAX inference server: [/max/python/max/serve](max/python/max/serve)
  (OpenAI-compatible endpoint)
- MAX model pipelines: [/max/python/max/pipelines](max/python/max/pipelines)
  (Python-based graphs)
- Code examples: [/max/examples](max/examples) + [/mojo/examples](mojo/examples)

This repo has two major branches:

- The [`main`](https://github.com/modular/modular/tree/main) branch, which is
in sync with the nightly build and subject to new bugs. Use this branch for
[contributions](./CONTRIBUTING.md), or if you [installed the nightly
build](https://docs.modular.com/max/packages).

- The [`stable`](https://github.com/modular/modular/tree/stable) branch, which
is in sync with the last stable released version of Mojo. Use the examples in
here if you [installed the stable
build](https://docs.modular.com/max/packages).

## Contribute

We accept contributions to the [Mojo standard library](./mojo), [MAX AI
kernels](./max/kernels), [MAX model
architectures](/max/python/max/pipelines/architectures), code examples, Mojo
docs, and more.

First, please read the [Contribution Guide](./CONTRIBUTING.md), and then refer
to the following documentation about how to develop in the repo:

- [`/max/docs`](/max/docs): Docs for developers working in the MAX framework codebase.
- [`/mojo/stdlib/docs`](/mojo/stdlib/docs): Docs for developers working in the
  Mojo standard library.

We also welcome your bug reports. If you have a bug, please [file an issue
here](https://github.com/modular/modular/issues/new/choose).

## News & Announcements

**[2026/2]** We announced that [BentoML is joining Modular][bentoml-joins].
We are committed to building in the open and will be extending our support
of open source AI with [Bento's own open project][bentoml-github]. Join us
[for an AMA February 17th][bentoml-joins-ama] to discuss our plans.

**[2026/1]** [Modular Platform 26.1][26.1] graduates the MAX Python API out of
experimental with PyTorch-like eager mode and model.compile() for production,
stabilizes the MAX LLM Book, and expands Apple silicon GPU support. Mojo gains
compile-time reflection, linear types, typed errors, and improved error messages
as it progresses toward 1.0.

**[2025/12]** [The Path to Mojo 1.0][mojo-1.0] was officially announced
with a planned release in H1 2026 and tons of details on what to expect.

**[2025/12]** We hosted our [Inside the MAX Framework Meetup][dec-meetup]
reintroducing the MAX framework and taking the community through upcoming
changes.

**[2025/11]** [Modular Platform 25.7][25.7] provides a fully open MAX Python
API, expanded hardware support for NVIDIA Grace superchips, improved Mojo GPU
programming experience, and much more.

**[2025/11]** We met with the community at
[PyTorch 2025 + the LLVM Developers' Meeting][pytorch-llvm] to solicit
community input into how the Modular platform can reduce fragmentation and
provide a unified AI stack.

**[2025/09]** [Modular raises \$250M][250-funding] to scale AI's unified compute
layer, bringing Modular's total raise to $380M at a $1.6B valuation.

**[2025/09]** [Modular Platform 25.6][25.6] delivers a unified compute layer
spanning from laptops to datacenter GPUs, with industry-leading throughput on
NVIDIA Blackwell (B200) and AMD MI355X.

**[2025/08]** [Modular Platform 25.5][25.5] introduces Large Scale Batch
Inference through a partnership with SF Compute + open source launch of the
MAX Graph API and more.

**[2025/08]** We hosted our [Los Altos Meetup][la-meetup] featuring talks from
Chris Lattner on democratizing AI compute and Inworld AI on production voice AI.

**[2025/06]** [AMD partnership announced][amd] — Modular Platform now generally
available across AMD's MI300 and MI325 GPU portfolio.

**[2025/06]** [Modular Hack Weekend][hack-weekend] brought developers together
to build custom kernels, model architectures, and PyTorch custom ops with
Mojo and MAX.

**[2025/05]** Over 100 engineers gathered at AGI House for our first
[GPU Kernel Hackathon][hackathon], featuring talks from Modular and
Anthropic engineers.

[bentoml-github]: https://github.com/bentoml/BentoML
[bentoml-joins-ama]: https://forum.modular.com/t/modular-has-acquired-bentoml-ask-us-anything/2706/1
[bentoml-joins]: https://www.modular.com/blog/bentoml-joins-modular
[ama-time]: https://www.inyourowntime.zone/2026-02-17_09.30_America.Los_Angeles
[26.1]: https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure
[mojo-1.0]: https://www.modular.com/blog/the-path-to-mojo-1-0
[dec-meetup]: https://www.youtube.com/live/WK5dVQ8vhbU?si=Fjde8j_50V4bwiAv
[25.7]: https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience
[250-funding]: https://www.modular.com/blog/modular-raises-250m-to-scale-ais-unified-compute-layer
[pytorch-llvm]: https://www.modular.com/blog/pytorch-and-llvm-in-2025-keeping-up-with-ai-innovation
[25.6]: https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
[25.5]: https://www.modular.com/blog/modular-platform-25-5
[la-meetup]: https://lu.ma/modular-aug-meetup
[amd]: https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus
[hack-weekend]: https://www.meetup.com/modular-meetup-group/events/308311461/
[hackathon]: https://www.modular.com/blog/modverse-48

---

## Community & Events

We host regular meetups, hackathons, and community calls. Join us!

| Channel               | Link                                            |
|-----------------------|-------------------------------------------------|
| 💬 Discord            | [discord.gg/modular][discord]                   |
| 💬 Forum              | [forum.modular.com][forum]                      |
| 📅 Meetup Group       | [meetup.com/modular-meetup-group][meetup-group] |
| 🎥 Community Meetings | Recordings on [YouTube][youtube]                |

**Upcoming events** will be posted on our [Meetup page][meetup-group] and
[Discord][discord].

[discord]: https://discord.gg/modular
[forum]: https://forum.modular.com/
[meetup-group]: https://www.meetup.com/modular-meetup-group/
[youtube]: https://www.youtube.com/@modularinc

## Contact us

If you'd like to chat with the team and other community members, please send a
message to our [Discord channel](https://discord.gg/modular) and [our
forum board](https://forum.modular.com/).

## License

This repository and its contributions are licensed under the Apache License
v2.0 with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)).
Modular, MAX and Mojo usage and distribution are licensed under the
[Modular Community License](https://www.modular.com/legal/community).

### Third party licenses

You are entirely responsible for checking and validating the licenses of
third parties (i.e. Huggingface) for related software and libraries that are downloaded.

## Thanks to our contributors

<a href="https://github.com/modular/modular/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modular/modular" />
</a>
