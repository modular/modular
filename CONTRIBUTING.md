# External Contribution Workflow for Modular Repo

We welcome and value contributions from the community! This document explains
the lifecycle of a pull request (PR) in the
[Modular repository](https://github.com/modular/modular), from submission
through review and final merge.

If you're looking for guidance on *how* to contribute, please refer to the
[CONTRIBUTING guide](mojo/CONTRIBUTING.md) for the Mojo standard library.

The MAX Kernel Library will also be open for contributions soon. For details
on contributing to this component, please see its dedicated
[CONTRIBUTING guide](max/kernels/CONTRIBUTING.md).

---

## 🧭 Overview of the Contribution Process

### Step 1: PR Triage and Initial Review

When a new pull request is opened by an external contributor, a Modular team
member will take an initial look and determine how to proceed. This may include:

- **Leaving the PR as-is** (e.g. if it's a draft).
- **Reviewing the PR directly**, especially if the changes are straightforward.
- **Assigning the PR** to a subject-matter expert on the appropriate team
  (Libraries, Kernels, Documentation etc.) for deeper review.

We aim to respond in a timely manner based on the time tables in the
[CONTRIBUTING guide](mojo/CONTRIBUTING.md#guidelines-for-review-time).

---

### Step 2: Internal Review and Syncing

Once a PR passes initial review and is progressing toward approval, a Modular
team member will sync it to our internal repository for further validation and
integration. This is done using an automated tool that mirrors your changes into
our internal environment.

💡 **Note**: This process is transparent to you as a contributor. You'll see a
bot (Modularbot) comment on your PR with status updates like:

- `Synced internally` - when your change has been synced internally into our
  repository
- `Merged internally` - when your change has been merged internally into our
  repository
- `Merged externally` - when your change has gone out with the latest nightly and
  is now available upstream in the `main` branch.

These messages help track the lifecycle of your contribution across our systems.

---

### Step 3: Review Feedback and Iteration

All feedback intended for you will be posted directly on the **external** pull
request. Internal discussions (e.g. security/privacy reviews or cross-team
coordination) may happen privately but won't affect your ability to contribute.
If we need changes from you, we'll leave clear comments with action items.

Once everything is approved and CI checks pass, we'll take care of the final
steps to get your PR merged.

---

## 🛠️ Behind the Scenes (FYI)

Here are a few implementation details that help us keep things running smoothly:

- We use a tool called [**Copybara**](https://github.com/google/copybara) to
  sync changes between internal and external repos.
- Your GitHub username and PR number are automatically preserved via commit
  metadata like:

    ```plaintext
    ORIGINAL_AUTHOR=username 12345678+username@users.noreply.github.com
    PUBLIC_PR_LINK=modularml/mojo#2439

    ```

## 🌙 Nightly Builds

The [Modular repository](https://github.com/modular/modular) is synced nightly
around 2 am ET. This means the `main` branch may lag slightly behind our
internal repository by up to 24 hours. At times, it may be longer in case of a
(blocking) release failure in our internal CI release workflows.

## 🙌 Thanks for Contributing

We deeply appreciate your interest in improving the Modular ecosystem. Whether
you're fixing typos, improving docs, or contributing core library features, your
input makes a difference.

If you have questions or need help, feel free to:

- Leave a comment on your pull request
- Join our community [forum](https://forum.modular.com/) and post a question

Let's build something great together!
