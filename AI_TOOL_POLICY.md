# Modular AI Tool Use Policy

> **This is a living document.** AI tools and community norms are evolving
> quickly. We expect to revise this policy over time. We’ll announce any
> major changes in our [community forum][forum] and you can see all
> changes in the [commit history][history].

## Quick Summary

- [**Label AI-assisted work**](#label-ai-assisted-work) using an
  `Assisted-by:` commit trailer/PR description
- [**Keep PRs small**](#keep-prs-small): AI lowers the cost of
  generating code, not reviewing it
- [**Write PR descriptions yourself**](#write-pr-descriptions-yourself)
  to ensure sufficient human involvement
- [**Keep humans in the loop**](#keep-humans-in-the-loop): The human
  author must review their changes before submitting AI-generated pull
  requests. Full AI-automation without human review is not currently
  permitted.
- [**Quality standards apply**](#quality-standards-apply): all relevant
  coding standards established within the Modular project still apply
  to AI-assisted contributions

For more context on contributing, [review the contributing guide.][contributing]

## Philosophy

At Modular, we believe AI tools are already fundamental to software
development, and we expect our contributors to
continue to embrace them. As [Chris Lattner has written][ceo-blog], AI coding is
automation of implementation: it lowers the cost of writing code, which makes
design, judgment, and human accountability *more* important, not less.

This policy exists not to limit the use of AI tools, but to ensure that when
AI generates contributions to this project, a skilled human is directing,
validating, and owning that work. The goal is better software and a healthier
community rather than policing how code gets written.

## Policy

Contributors are free to use whatever tools they like to craft their
contributions, but **there must be a human in the loop**. Contributors must
read and review all AI-generated code or text before asking other project
members to review it. The contributor is always the author and is fully
accountable for their contribution — for its correctness, design quality, and
long-term maintainability. **AI expands your capabilities; it does not
outsource your judgment.** Contributors should be sufficiently confident that
their contribution is high enough quality that requesting a review is a good
use of scarce maintainer time, and contributors should be **able to answer questions
about their work** during review.

We expect that newer contributors will be less confident in their
contributions. Our guidance to them is to **start with small contributions**
they can fully understand, to build confidence over time. We intend to be a
welcoming community that helps contributors grow their expertise, but learning
involves taking small steps, getting feedback, and iterating. Passing
maintainer feedback directly to an LLM and resubmitting doesn't help anyone
grow, and does not sustain our community.

### Label AI-assisted Work

Contributors are expected to **be transparent and label contributions that
contain substantial amounts of AI-generated content**. This policy on labeling
is intended to facilitate code review, not to track which parts of the
codebase were generated. Note AI tool usage in your pull request description,
commit message, or wherever authorship is normally indicated. For example,
use a commit message trailer:

```text
Assisted-by: AI
```

This transparency helps the community develop best practices and understand
the role of these new tools.

This policy applies to, but is not limited to, the following kinds of
contributions:

- Code and doc changes in a pull request
- RFCs or design proposals
- Issues or security vulnerabilities
- Comments and feedback on pull requests

### Keep PRs Small

AI tools make it tempting to generate large, wide-ranging changes in a single
PR. We encourage contributors to resist that temptation and prefer small,
focused pull requests even when the tooling makes a larger change feel easy
to produce.

As a general rule, try to keep PRs under 100 lines of code whenever possible.
For more details about PR size, review [contributing.md][contributing-pr-size].

A focused PR does one thing completely: scoped to a single bug,
feature, or improvement, with a coherent set of changes and a short diff.
A PR that touches one thing is faster to review, easier to reason about,
and more likely to be accepted quickly.

AI shifts the economics of software development: generating a large patch now
costs the author very little, but the reviewer's cost is unchanged. A 500-line
PR that could have been three 150-line PRs doesn't become easier to review
just because it was produced quickly. Keeping PRs small and focused is one of
the most effective ways to ensure your contribution is worth the review time
it requires - and to improve the likelihood that it will be approved.

### Write PR Descriptions Yourself

To ensure sufficient self-review and understanding of the work, contributors
should write PR descriptions themselves (using AI tools for translation or
copy-editing is fine).

The description should explain the motivation,
implementation approach, expected impact, and any open questions or
uncertainties to the same extent as a contribution written without AI
assistance.

#### Structuring PR Descriptions

Please structure PR descriptions for easy review. While it's important to highlight
changes and provide strong explanations, reviewing long PR descriptions also takes
time, just as it does for PRs with significant lines of code. Bullet points or use
of other structure in descriptions are recommended.

### Keep Humans in the Loop

This policy prohibits agents that take action in our digital spaces without
human approval, for example, bots that automatically open or comment on
GitHub issues and PRs. However, an opt-in tool that **keeps a
human in the loop** is acceptable. For example, using an LLM to generate
documentation, which a contributor then manually reviews for correctness,
edits, and posts as a PR, is an approved use of tools under this policy.

### Quality Standards Apply

All contributions must meet Modular's standards for code quality, testing, and
correctness — regardless of how they were produced. See the
[contributing guide][contributing] for requirements such as formatting, testing,
and code review expectations.

## Extractive Contributions & Rationale

The reason for our human-in-the-loop policy is that reviewing patches, PRs,
RFCs, and comments takes real maintainer time and energy. Sending the
unreviewed output of an AI to open source project maintainers *extracts* work
from them in the form of design and code review. These can be considered
"extractive contributions."

Our **golden rule** is that a contribution should be worth more to the project
than the time it takes to review it. Nadia Eghbal captures this well in her
book [*Working in Public*][public]:

> "When attention is being appropriated, producers need to weigh the costs and
> benefits of the transaction. To assess whether the appropriation of attention
> is net-positive, it's useful to distinguish between *extractive* and
> *non-extractive* contributions. Extractive contributions are those where the
> marginal cost of reviewing and merging that contribution is greater than the
> marginal benefit to the project's producers. In the case of a code
> contribution, it might be a pull request that's too complex or unwieldy to
> review, given the potential upside." — Nadia Eghbal

Before LLMs, maintainers would often review any contribution because submitting
one signaled genuine interest from a potential long-term contributor. AI tools
change that signal. They shift effort from the implementor to the reviewer, and
our policy exists to protect the maintainer time that keeps this project alive.

Reviewing changes from new contributors is how we grow the next generation of
maintainers. We want this project to be open to contributors who are willing
to invest time and effort to learn, expanding our contributor base is
what sustains the project over the long term.

## Handling Violations

If a maintainer judges that a contribution doesn't comply with this policy,
they should paste the following response to request changes:

```text
This PR doesn't appear to comply with our policy on AI-generated content,
and requires additional justification for why it is valuable enough to the
project for us to review it. Please see our AI tool use policy:
https://github.com/modular/modular/blob/main/oss/modular/AI_TOOL_POLICY.md
```

The best ways to make a change less extractive are to reduce its size or
complexity, or to increase its usefulness to the community. These factors
cannot be weighed objectively. Our policy leaves this determination to the
maintainers doing the work of sustaining the project.

When a GitHub issue or PR is clearly off-track, maintainers should apply the
[`extractive`][extractive-label] label to help other reviewers prioritize
their time.

If a contributor fails to make their change meaningfully less extractive,
maintainers should escalate to the relevant moderation team per our
[Code of Conduct][coconduct] to lock the conversation.

## Copyright

AI systems raise unsettled questions around copyright. Our position is
consistent with our broader copyright policy: contributors are responsible for
ensuring they have the right to contribute code under the terms of our
license. Using AI tools to regenerate copyrighted material does not remove the
copyright. Contributors are responsible for ensuring that such material does
not appear in their contributions. Contributions found to violate this will be
removed like any other offending contribution.

## References

This policy was informed by policies and experiences in other communities:

- [LLVM AI Tool Policy][llvm] (accessed 2026-02-20, Apache License 2.0 with
  LLVM Exceptions): The structure and significant portions of the text of this
  policy are adapted from the LLVM project's AI Tool Policy.
- [Fedora Council Policy Proposal: Policy on AI-Assisted Contributions][fedora]:
  Some text was drawn from the Fedora project policy proposal, which is
  licensed under the [Creative Commons Attribution 4.0 International
  License][cca]. This link serves as attribution.
- [Rust draft policy on burdensome PRs][rust-burdensome]
- [Seth Larson's post][security-slop] on slop security reports in the Python
  ecosystem
- [QEMU bans use of AI content generators][qemu-ban]
- [Slop is the new name for unwanted AI-generated content][ai-slop]

<!-- Link references -->
[contributing-pr-size]: https://github.com/modular/modular/blob/main/max/CONTRIBUTING.md#about-pull-request-sizes
[contributing]: https://github.com/modular/modular/blob/main/CONTRIBUTING.md
[forum]: https://forum.modular.com/
[history]: https://github.com/modular/modular/commits/main/oss/modular/AI_TOOL_POLICY.md
[ceo-blog]: https://www.modular.com/blog/the-claude-c-compiler-what-it-reveals-about-the-future-of-software
[public]: https://press.stripe.com/working-in-public
[extractive-label]: https://github.com/modular/modular/issues?q=label%3Aextractive
[coconduct]: ./CODE_OF_CONDUCT.md
[llvm]: https://llvm.org/docs/AIToolPolicy.html
[fedora]: https://communityblog.fedoraproject.org/council-policy-proposal-policy-on-ai-assisted-contributions/
[cca]: https://creativecommons.org/licenses/by/4.0/
[rust-burdensome]: https://github.com/rust-lang/compiler-team/issues/893
[security-slop]: https://sethmlarson.dev/slop-security-reports
[qemu-ban]: https://www.qemu.org/docs/master/devel/code-provenance.html#use-of-ai-content-generators
[ai-slop]: https://simonwillison.net/2024/May/8/slop/
