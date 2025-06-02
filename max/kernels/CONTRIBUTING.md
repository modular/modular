# Contributing to the MAX AI kernels

Thank you for your interest in contributing to the MAX AI kernels! We’re excited
to welcome external contributors to help build high-performance CPU and GPU
kernels in Mojo.

## ✅ We are accepting contributions

The MAX AI kernels power key components of MAX and Mojo, and contributions here
can have a meaningful impact across the AI ecosystem. Before jumping in, please
take a moment to review our contribution guidelines and processes.

## Contribution guidelines

We welcome contributions across a wide range of functionality, but **if you’re
proposing a change that could affect the performance of the core kernel
library**, we ask that you first go through our **proposal process**.

### Proposals

If you’re interested in making a significant change (one that doesn’t fall into
the [list of changes we accept](#Changes-we-accept) below), your first step is a
written proposal. The proposal process:

- Ensures feedback from the widest possible set of community members
- Maintains an audit trail of important design decisions
- Provides rationale for changes that impact long-term performance or
architecture

**To submit a proposal, create a GitHub Pull Request that adds a document to the
[`modular/mojo/proposals`](https://github.com/modular/modular/tree/main/mojo/proposals)
directory.**

Contributors are encouraged to react with a 👍 to proposal PRs if they support
the high-level direction. Proposals are reviewed and assigned to MAX AI kernels
leads. A proposal can be merged once the lead approves, all blocking issues are
resolved, and relevant decisions are incorporated. If a lead chooses to defer or
reject the proposal, they will explain the reasoning and close the PR. This
process is inspired by successful practices in other open source communities.

### Changes we accept

We welcome contributions across all hardware platforms and are particularly
interested in Blackwell, Hopper, MI3xx, and other data center GPUs.

In particular, we welcome contributions of the following kernels:

- Batched matrix multiplication (BMM)
- Multi-head latent attention (MLA)
- Mixture of experts (MOE)
- Non-maximum suppression (NMS)
- Grouped matrix multiplication
- 2D convolutions
- General matrix-vector multiply (GEMV) on Hopper

## Submitting bugs

Reporting issues is a great way to contribute to the project.

Keep in mind that bugs with clear reproducible steps and/or test cases, and
well-written documentation will be considered a higher priority.

Also, before opening a new issue, take a moment to search through the already
submitted issues to avoid creating duplicate issues for the maintainers to
address.

### Writing high-quality bug descriptions

We encourage you to provide as much information about the issue as practical.
The more details you provide, the faster we can resolve the issue. The following
is a template of the information that should accompany every submitted issue.

### Issue template

- **Summary:** a descriptive summary of the issue.
- **Description:** a detailed account of the bug, including what was expected
and what occurred.
- **Environment details**
  - Mojo compiler version
  - Operating system version
  - Hardware specifications
- **Severity/frequency**: an assessment of the impact, ranging from
inconvenience to a blocker.
- **Steps to reproduce:** a simple procedure to reproduce the bug you
encountered.

## Submitting pull requests

You can use a pull request to propose a change or bug fix. This page gives an
overview of the process.

**Note:** Pull requests should be submitted against the `main` branch, which
represents the most recent nightly build.

### **Pull request process**

#### First-time checklist

Before you start your first pull request, please complete this checklist:

- Read this entire contributor guide.
- Read the [Code of
Conduct](https://github.com/modular/modular/blob/main/CODE_OF_CONDUCT.md).

#### Evaluate and get buy-in on the change

We want to be sure that you spend your time efficiently and prepare changes that
aren’t controversial and get stuck in long rounds of reviews. See [the section
on changes we accept](#changes-we-accept) for more details.

#### Fork and clone the repo

Go to the [modular repo](https://github.com/modular/modular) and click the fork
button:

![](https://github.com/modular/modular/raw/main/mojo/stdlib/docs/images/create-fork.png)

Clone your forked repo locally with the command:

```bash
git clone git@github.com:[your-username]/modular.git
cd modular
```

Add the upstream remote and fetch it:

```bash
git remote add upstream git@github.com:modular/modular.git
git fetch upstream
```

#### Branching off main

Make sure to branch off `main` to work on your PR:

```bash
git checkout main
git checkout -b my-fix-pr
```

You should periodically make sure you've synced the latest changes, especially
before raising a PR:

```bash
git fetch upstream
git rebase upstream/main
```

#### Getting the nightly Mojo compiler

Now that you're on the main branch, you need to install the latest nightly
build.

Create a new Mojo project using [Pixi](https://pixi.sh/latest/):

```bash
pixi init myproj -c "https://conda.modular.com/max-nightly" -c
"https://repo.prefix.dev/modular-community" -c "conda-forge"
```

If you're [using conda](https://docs.modular.com/magic/conda), add
the `https://conda.modular.com/max-nightly/` channel to
your `environment.yaml` file. For example:

```yaml
[project]
name = "Mojo nightly example"
channels = ["conda-forge", "https://conda.modular.com/max-nightly/"]
platforms = ["osx-arm64", "linux-aarch64", "linux-64"]

[dependencies]
max = "*"
```

#### Mojo nightly vscode extension

Install the [Mojo nightly VS Code
extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo-nightly):

![](https://github.com/modular/modular/raw/main/mojo/stdlib/docs/images/nightly-extension.png)

You can only have one Mojo extension enabled at a time, remember to switch back
when using the stable release!

#### Create a pull request

If your change is one of the improvements described above or has been discussed
and agreed upon by the project maintainers, please create a pull request into
the `main` branch.

First, push your changes:

```bash
git push -u origin my-fix-pr
```

You'll see a link to create a PR:

```bash
remote: Create a pull request for 'my-fix-pr' on GitHub by visiting:
remote:      https://github.com/[your-username]/max/pull/new/my-fix-pr

```

It should automatically set the base branch to the upstream `origin/main`, but
if it doesn't, you can set it manually.

Now fill out the details:

- A short commit title describing the change.
- A detailed commit description that includes rationalization for the change
and/or explanation of the problem that it solves, with a link to any relevant
GitHub issues.

**Note:** Accepted changes will generally show up in the release build (or on
the website) for the next *major* release.

Thank you for your contributions! ❤️

#### Signing the CLA

For each pull request, we require that you certify that you wrote the change or
otherwise have the right to pass it on as an open-source patch by signing [our
contributor license agreement](https://github.com/modular/cla/blob/main/CLA.md).
You can sign the CLA by posting a comment on your pull request that says, "I
have read the CLA Document and I hereby sign the CLA." The `github-actions` bot
will leave a comment on your pull request reminding you to complete this step.

## Guidelines for Review Time

### Pull Request (PR) Review Timeline

#### Initial Review

Maintainers will provide an initial review or feedback within 3 weeks of the PR
submission. At times, it may be significantly quicker, but it depends on a
variety of factors.

#### Subsequent Reviews

Once a contributor addresses feedback, maintainers will review updates as soon
as they can, typically within 5 business days.

### Issue Triage Timeline

#### New Issues

Maintainers will label and acknowledge new issues within 10 days of the issue
submission.

### Proposal Review Timeline

Proposals require more time for the team to review, discuss, and ensure they
align with the overall strategy and vision for the MAX AI kernels. These will be
discussed in the team's weekly design meetings internally, and feedback will be
communicated back on the relevant proposal. As a team, we'll ensure they are
reviewed and discussed within 6 weeks of submission.

### Exceptions

While we strive our best to adhere to these timelines, there may be occasional
delays due to any of the following:

- High volume of contributions.
- Maintainers' availability (e.g., holidays, team events).
- Complex issues or PRs requiring extended discussion (these may get deferred to
the team's weekly design discussion meetings).

Note that just because a pull request has been reviewed does not necessarily
mean it will be able to be merged internally immediately. This could be due to a
variety of reasons, such as:

- Mojo compiler bugs. These take time to find a minimal reproducer, file an
issue with the compiler team, and then get prioritized and fixed.
- Internal bugs that get exposed due to a changeset.
- Massive refactorings due to an external changeset. These also take time to fix
- remember, we have the largest Mojo codebase in the world internally.

If delays occur, we'll provide status updates in the relevant thread (pull
request or GitHub issue).

#### How You Can Help

To ensure quicker reviews:

- **Ensure your PR is small and focused.** See the [pull request size section]
 for more info.
- Write a good commit message/PR summary outlining the motivation and describing
the changes. In the near future, we'll provide a pull request template to
clarify this further.
- Use descriptive titles and comments for clarity.
- Code-review other contributors' pull requests and help each other.

[pull request size section]: https://github.com/modular/modular/blob/main/mojo/CONTRIBUTING.md#about-pull-request-sizes
