# Modular contributor guide

Thank you for your interest in contributing to this repository!

This page explains the overall process to create a pull request (PR), from
forking the repo all the way through review and final merge.

> [!NOTE]
> We accept code contributions to the [Mojo standard library](./mojo), [MAX
> AI kernels](./max/kernels), [code examples](./examples), and
> [Mojo docs](./mojo/docs), but currently not to
> other parts of the repository.

## Submitting bugs

Reporting issues is a great way to contribute to the project.

Before opening a new issue, take a moment to search through the already
[submitted issues](https://github.com/modular/modular/issues) to avoid creating
duplicate issues for the maintainers to address.

### Writing high-quality bug descriptions

Bugs with a reproducible test case and well-written descriptions will be
considered a higher priority.

We encourage you to provide as much information about the issue as practical.
The more details you provide, the faster we can resolve the issue. The following
is a template of the information that should accompany every submitted issue.

#### Issue template

- **Summary.** A descriptive summary of the issue.
- **Description.** A detailed account of the bug, including what was expected
  and what occurred.
- **Environment details.**
  - MAX or Mojo version (run `max --version` or `mojo --version`)
  - Operating system version
  - Hardware specifications
- **Severity/frequency.** An assessment of the impact ranging from inconvenience
  to a blocker.

## Contributing changes

Before you start your first pull request, please complete this checklist:

- Read this entire contributor guide.
- Read the [Code of Conduct](../CODE_OF_CONDUCT.md).

### Step 1: Evaluate and get buy-in on the change

First, consider that several parts of this repository currently do not accept
contributions. You should refer to the README or CONTRIBUTING file nearest the
code you're interested in.

We also want to be sure that you spend your time efficiently and prepare
changes that aren’t controversial and get stuck in long rounds of reviews. So
if the change is non-trivial, please submit an issue or write a proposal, as
described in the corresponding sections.

For example, we accept contributions to the following sections where you can
find specific contribution guidelines:

- [Mojo standard library](mojo/CONTRIBUTING.md)
- [MAX AI kernels](max/kernels/CONTRIBUTING.md)
- [Code examples](examples#contributing)
- [Mojo documentation](mojo/docs#contributing)

### Step 2: Create a pull request

If you're experienced with GitHub, here's the basic process:

1. Fork this repo.

2. Create a branch from `main`.

   If you're contributing to the Mojo standard library, see the
   [Mojo standard library developer guide](mojo/stdlib/docs/development.md).

3. Create a PR into the `main` branch of this repo.

4. Skip to [Step 3: PR triage and review](#step-3-pr-triage-and-review).

#### Pull request walkthrough

For more specifics, here's a detailed walkthrough of the process to create a
pull request:

1. Fork and clone this repo:

    Go to the [modular repo home](https://github.com/modular/modular) and click
    the **Fork** button at the top.

    Your fork will be accessible at `https://github.com/<your-username>/modular`.

    Clone your forked repo to your computer:

    ```bash
    git clone git@github.com:<your-username>/modular.git
    cd modular
    ```

    To clarify, you're working with three repo entities:

    - This repo (`https://github.com/modular/modular`) is known as the upstream
      repo. In Git terminology, it's the *upstream remote*.
    - Your fork on GitHub is known as *origin* (also remote).
    - Your local clone is stored on our computer.

    Because a fork can diverge from the upstream repo it was forked from, it is
    crucial to configure our local clone to track upstream changes:

    ```bash
    git remote add upstream git@github.com:modular/modular.git
    ```

    Then sync your fork to the latest code from upstream:

    ```bash
    git pull --rebase upstream
    ```

2. Create a branch off `main` to work on your change:

    ```bash
    git checkout -b my-fix
    ```

    Now start your work on the repo! If you're contributing to the Mojo
    standard library, see the [Mojo standard library developer
    guide](mojo/stdlib/docs/development.md).

    Although not necessary right now, you should periodically make sure you have
    the latest code, especially right before you create the pull request:

    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

3. Create a pull request:

    When you're code is ready, create a pull request into the `main` branch.

    First push the local changes to your origin on GitHub:

    ```bash
    git push -u origin my-fix
    ```

    You'll see a link to create a PR:

    ```plaintext
    remote: Create a pull request for 'my-fix' on GitHub by visiting:
    remote:      https://github.com/[your-username]/modular/pull/new/my-fix
    ```

    You can open that URL or visit your fork on GitHub and click **Contribute** to
    start a pull request.

    GitHub should automatically set the base repository to `modular/modular`
    and the base (branch) to `main`. If not, you can select it from the drop-down.
    Then click **Create pull request**.

    Now fill out the pull request details in the GitHub UI:

    - Add a short commit title describing the change.
    - Add a detailed commit description that includes rationalization for the change
      and/or explanation of the problem that it solves, with a link to any relevant
      GitHub issues.
    - Add a `Signed-off-by` line for the "Developer Certificate of
      Origin"—see the section below about [signing your work](#signing-your-work).

    Click **Create pull request**.

### Step 3: PR triage and review

A Modular team member will take an initial look the the pull request and
determine how to proceed. This may include:

- **Leaving the PR as-is** (e.g. if it's a draft).
- **Reviewing the PR directly**, especially if the changes are straightforward.
- **Assigning the PR** to a subject-matter expert on the appropriate team
  (Libraries, Kernels, Documentation etc.) for deeper review.

We aim to respond in a timely manner based on the time tables in the
[guidelines for review time](#guidelines-for-review-time), below.

### Step 4: Internal review and syncing

Once a PR passes initial review and is progressing toward approval, a Modular
team member will sync it to our internal repository for further validation and
integration. This is done using an automated tool that mirrors your changes into
our internal environment.

This process is transparent to you as a contributor. You'll see a bot
(Modularbot) comment on your PR with status updates like:

- `Synced internally` - when your change has been synced internally into our
  repository
- `Merged internally` - when your change has been merged internally into our
  repository
- `Merged externally` - when your change has gone out with the latest nightly and
  is now available upstream in the `main` branch.

These messages help track the lifecycle of your contribution across our systems.

### Step 5: Review feedback and iteration

All feedback intended for you will be posted directly on the **external** pull
request. Internal discussions (e.g. security/privacy reviews or cross-team
coordination) may happen privately but won't affect your ability to contribute.
If we need changes from you, we'll leave clear comments with action items.

Once everything is approved and CI checks pass, we'll take care of the final
steps to get your PR merged.

Merged changes will generally show up in the the next nightly build (or docs
website), a day or two after it's merged.

## Signing your work

For each pull request, we require that you certify that you wrote the change or
otherwise have the right to pass it on as an open-source patch by adding a line
at the end of your commit description message in the form of:

`Signed-off-by: Jamie Smith <jamie.smith@example.com>`

You must use your real name to contribute (no pseudonyms or anonymous
contributions). If you set your `user.name` and `user.email` git configs, you
can sign your commit automatically with `git commit -s`.

Doing so serves as a digital signature in agreement to the following
Developer Certificate of Origin (DCO):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Guidelines for review time

1. Pull Request (PR) Review Timeline

   Initial Review:
    - Maintainers will provide an initial review or feedback within 3 weeks of
      the PR submission.  At times, it may be significantly quicker, but it
      depends on a variety of factors.

   Subsequent Reviews:
    - Once a contributor addresses feedback, maintainers will review updates as
      soon as they can, typically within 5 business days.

1. Issue Triage Timeline

   New Issues:
   - Maintainers will label and acknowledge new issues within 10 days of the
      issue submission.

1. Proposals

   - Proposals take more time for the team to review, discuss, and make sure this
    is in line with the overall strategy and vision for the standard library.
    These will get discussed in the team's weekly design meetings internally and
    feedback will be communicated back on the relevant proposal.  As a team, we'll
    ensure these get reviewed and discussed within 6 weeks of submission.

### Exceptions

While we strive our best to adhere to these timelines, there may be occasional
delays due any of the following:

- High volume of contributions.
- Maintainers' availability (e.g. holidays, team events).
- Complex issues or PRs requiring extended discussion (these may get deferred to
  the team's weekly design discussion meetings).

Note that just because a pull request has been reviewed does not necessarily
mean it will be able to be merged internally immediately.  This could be due to a
variety of reasons, such as:

- Mojo compiler bugs.  These take time to find a minimal reproducer, file an
  issue with the compiler team, and then get prioritized and fixed.
- Internal bugs that get exposed due to a changeset.
- Massive refactorings due to an external changeset.  These also take time to
  fix - remember, we have the largest Mojo codebase in the world internally.

If delays occur, we'll provide status updates in the relevant thread (pull
request or GitHub issue).  Please bear with us as Mojo is an early language.
We look forward to working together with you in making Mojo better for everyone!

### How you can help

To ensure quicker reviews:

- **Ensure your PR is small and focused.** See the [pull request size section](#about-pull-request-sizes)
  for more info.
- Write a good commit message/PR summary outlining the motivation and describing
  the changes.  In the near future, we'll provide a pull request template to
  clarify this further.
- Use descriptive titles and comments for clarity.
- Code-review other contributor pull requests and help each other.

## Behind the scenes (FYI)

Here are a few implementation details that help us keep things running smoothly:

- We use a tool called [**Copybara**](https://github.com/google/copybara) to
  sync changes between internal and external repos.

- Your GitHub username and PR number are automatically preserved via commit
  metadata like:

    ```plaintext
    ORIGINAL_AUTHOR=username 12345678+username@users.noreply.github.com
    PUBLIC_PR_LINK=modularml/mojo#2439
    ```

- This repo is synced nightly with Modular's internal repo around 2 am ET
almost every day. This means the `main` branch may lag slightly behind our
internal repository by up to 24 hours. At times, it may be longer in case of a
(blocking) release failure in our internal CI release workflows.

## 🙌 Thanks for contributing

We deeply appreciate your interest in improving the Modular ecosystem. Whether
you're fixing typos, improving docs, or contributing core library features, your
input makes a difference.

If you have questions or need help, feel free to:

- Leave a comment on your pull request
- Join our community [forum](https://forum.modular.com/) and post a question

Let's build something great together!
