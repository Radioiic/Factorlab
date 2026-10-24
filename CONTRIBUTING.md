# Contributing to Factorlab

Thank you for your interest in Factorlab.

## Issues and pull requests

**Factorlab does not accept pull requests. Issues are welcome.**

Use an issue for:

- a bug report;
- an error in the documentation;
- a question about the method, the results or the license;
- a proposal for a change.

Describe a change in words. Do not paste code for Factorlab into an issue. The maintainer
writes all code of Factorlab. Read [ROADMAP.md](ROADMAP.md) before you propose a change.

The rules of behavior are in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). To report a problem of
behavior, see [Report a problem](CODE_OF_CONDUCT.md#report-a-problem).

## Report a bug

Use the bug report form of the issues. It asks for:

1. the version (the file `VERSION`) or the commit (`git rev-parse --short HEAD`);
2. your system (macOS or Linux, and the version) and your Python version;
3. the command that you ran;
4. the full message or the last lines of `engine/logs/update.log`;
5. the result that you expected.

Remove your user name and your private paths from each message and each log before you
send it. For example, replace `/Users/name/` or `/home/name/` with `~/`.

Do not attach downloaded market data or exported dashboard data. The terms of the data
providers do not allow you to share it (see
[Data sources and product rules](README.md#data-sources-and-product-rules)).

## Report a documentation error or propose a change

Use the form "Documentation error or change proposal" of the issues. Give the document and
section, or the command, page or function, and describe the change in words.

## Report a vulnerability

Do not open an issue for a vulnerability. Use the private reporting form of GitHub.
[SECURITY.md](SECURITY.md#report-a-vulnerability) gives the steps.

## Run the tests

```bash
./setup.sh --test                                  # build, then the full test suite
engine/.venv/bin/python -B engine/run_tests.py     # the full suite only
```

The runner prints one line for each suite: PASS, SKIP or FAIL. A suite is SKIP when all
its tests need data that is not on disk. The last line gives the number of passed,
skipped and failed suites.

The GitHub Actions workflow `.github/workflows/tests.yml` has no market data. It has three
jobs:

- `tests`: it builds the library catalog, runs the code checks (ruff, bandit, mypy) and
  runs all the suites under coverage. It runs for each push to `main`, each pull request,
  each version tag `v*` and on request;
- `demo data`: it builds the synthetic demo data set (`engine/update.py --demo`) and runs
  all the suites with it under coverage. A suite that shows SKIP fails the job. A coverage
  total below 84 % also fails the job. On a GitHub-hosted runner the build alone takes
  about 30 minutes. The job runs each Monday, for each version tag `v*` and on request;
- `pip-audit`: it checks the pinned packages for known vulnerabilities. It runs when the
  `tests` job runs, and also each Monday.

`.github/dependabot.yml` lets Dependabot check the pinned packages and the actions each
month. It opens no pull requests.

[README.md](README.md#tests) gives the commands of the code checks on your computer.

## Maintenance

Radioiic maintains Factorlab and decides each issue.

The maintainer reads new issues at least one time in 14 days. An issue without an answer
is not a refusal.

The maintainer puts one or more of these labels on an issue:

| Label | Meaning |
|---|---|
| `bug` | Factorlab does not do what its documents say. |
| `documentation` | A document is wrong or not clear. |
| `question` | A question about the method, the results or the use. |
| `enhancement` | A proposal for a new function or a change. |
| `duplicate` | Another issue has the same subject. The maintainer closes this issue. |
| `invalid` | The issue is not about Factorlab, or it has not enough data. The maintainer closes the issue. |

The rules for the code of Factorlab:

- **Tests.** Each new function has tests in `engine/test_*.py` or `library/tests`. The
  full suite passes before each push to `main`.
- **Coding standard.** PEP 8 and the ruff rules in `pyproject.toml`.
- **Documents.** README.md describes each function. A change to a function changes the
  documents in the same commit.
- **Version.** The file `VERSION` holds the version, in the form MAJOR.MINOR.PATCH
  (Semantic Versioning 2.0.0).
- **Changes.** Each change that a user can see gets one item in
  [CHANGELOG.md](CHANGELOG.md), under `[Unreleased]`, in the same commit. Write each item
  on one line.
- **Releases.** A release follows the [release steps](#release-steps).

### Release steps

1. Set the new version X.Y.Z in `VERSION`, `library/pyproject.toml` and `CITATION.cff`
   (`version`). Set the UTC date of the release in `date-released` of `CITATION.cff`.
2. Set the new version in the version badge of `README.md`. Set it in the version examples
   of `README.md` and `engine/README_AUTOMATION.md`. Set it in the placeholders of the
   issue forms.
3. In `CHANGELOG.md`, move the items under `[Unreleased]` to a new section
   `## [X.Y.Z] - YYYY-MM-DD` (the same UTC date).
4. At the end of `CHANGELOG.md`, set the link `[Unreleased]` to `compare/vX.Y.Z...HEAD`.
   Add the link `[X.Y.Z]` to `releases/tag/vX.Y.Z`.
5. Run the full test suite. `test_version.py` checks the steps 1 to 4.
6. Commit the change. Tag the commit `vX.Y.Z`. Push the commit and the tag:

   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

   The push of the tag starts the test workflow with the demo data job.
7. Make the GitHub release with the text of the new section:

   ```bash
   engine/.venv/bin/python engine/version.py --changelog | gh release create vX.Y.Z --verify-tag --title "Factorlab X.Y.Z" --notes-file -
   ```

## License

The code of Factorlab is under the [MIT License](LICENSE). The market data keeps its own
terms (see [Data sources and product rules](README.md#data-sources-and-product-rules)).
For a question about the license, open an issue with the form "Question".
