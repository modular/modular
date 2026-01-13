# MAX contributor guide

Welcome to the MAX community! 🧑‍🚀 We’re very excited that you’re interested in
contributing to the project. To help you get started and ensure a smooth
process, we’ve put together this contributor guide.

There are many ways to contribute to the project, from joining our
[forum](https://forum.modular.com/) or
[Discord community](https://www.discord.gg/modular), to filing bugs, to
contributing documentation, examples, or code.

## Contributing to the MAX framework and models

To ensure a streamlined process, contributors are encouraged to focus on
enhancements, bug fixes, and optimizations aligned with the overarching goals
of MAX. These guidelines aim to facilitate a collaborative
environment where contributors and the Modular team can work together
effectively toward the continued improvement of MAX.

For technical details on developing for MAX and models, see the following
document:

- [Developing the MAX framework](./development.md) covers building,
  testing, and other information you’ll need to work in the MAX framework.

### Changes we *accept*

These changes are uncontroversial, easier to review, and more likely to be
accepted:

- Well-documented bug fixes submitted with code reproducing the issue in a test
  or benchmark.
- Performance improvements that don’t sacrifice code readability or
  maintainability and are accompanied by benchmarks.
- Improvements to API documentation.
- Improvements to the test coverage.
- Changes that address security vulnerabilities.

### Changes we *avoid*

Changes that don’t align with our vision and roadmap are unlikely to be
accepted. For example:

- Changes that do not align with the core principles of
  the MAX framework. When in doubt, feel free to reach out to the Modular team
  or community in our [forum](https://forum.modular.com/) or
  [Discord](https://www.discord.gg/modular) to see if a potential change would
  be well-received.
- Code without tests—especially for core primitives.
- Changes that break existing API, models, or implicit behavior semantics.
- Changes where the contributors’ favorite feature or system isn’t being used
  and they submit a change unilaterally switching the project to use it. For
  example, the contributor doesn’t like Bazel as a build system and submits a PR
  changing the repository to use their favorite build system.
- Adding support for esoteric platforms.
- Adding dependencies to the code base.
- Broad formatting or refactoring changes.
- Changes that need broad community consensus.
- Changes if contributors are not responsive to review comments.

## About pull request sizes

We ask that contributors make pull requests as small as possible. When
you are opening a pull request, check the number of lines modified in GitHub.
The smaller the better (but don't exclude the tests or docstrings). If your
pull request is over 100 lines, please try to split it into multiple pull
requests. If you make them independent, it's even better as no synchronization
will be needed for the merge.

This guideline is here for the following reasons:

- **Higher quality reviews**: It is much easier to spot a bug in a few lines
than in 1000 lines.
- **Faster overall review**: Reviewers, to approve a pull request, need to
understand every line and understand how it fits into your overall change.
They also need to go back and forth between files and functions to understand
the flow of the code. This is exponentially hard as there are more lines in the code.
- **Avoiding blocking changes that are valid**: In a huge pull request, it's
likely that some changes are valid and some need to be reworked/discussed. If all
the changes are in the same pull request, then the valid changes will be blocked
until all discussions have been resolved.
- **Reducing the number of git conflicts**: Bigger pull request means slower reviews,
thus means that the pull request will be open longer and will have more git conflicts
to be resolved before being merged.
- **Parallel processing**: All programmers like to parallelize. Well, reviewers also
like to parallelize code reviews to merge your code faster. If you open two pull
requests that are independent, then two reviewers will be able to work on your
code.
- **Finding the time for a code review**: Doing a code review often requires
that the code is reviewed in one go, as it's hard to remember functions and code
logic from one review session to another. Thus a big pull request will require
the reviewer to allocate a big chunk of time to do the code review, which is not
always possible and might delay the review and merge of your pull request
for multiple days.

Smaller pull requests means less work for the maintainers and faster reviews
and merges for the contributors. It's a win-win!

To help break apart big projects into smaller PRs, try using the
[stack-pr](https://github.com/modular/stack-pr) tool, which simplifies
the process of creating several PRs in a dependency chain.

## Submitting pull requests

For details about how to submit a pull request, see the repo's
[primary contributing guide](../CONTRIBUTING.md).

Thank you for your contributions! ❤️
