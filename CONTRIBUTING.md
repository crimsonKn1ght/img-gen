# Contributing to img-gen

Thank you for considering a contribution. Contributions of all sizes are welcome,
including documentation improvements, bug reports, tests, accessibility fixes, and
focused feature additions.

## Find something to work on

- Look for issues labelled `good first issue` if this is your first contribution.
- Look for `help wanted` issues when the maintainer has requested community help.
- Comment on an issue before starting work so effort is not duplicated.
- Open a feature request before implementing a substantial or potentially breaking
  change.

If you want to work on an existing issue, leave a short comment describing your
intended approach. The maintainer can then confirm the scope or assign the issue.

## Development setup

1. Fork the repository on GitHub.
2. Clone your fork and enter the project directory:

   ```bash
   git clone https://github.com/YOUR-USERNAME/img-gen.git
   cd img-gen
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   streamlit run streamlit_app.py
   ```

The first image-generation run downloads the selected model weights. This can take
considerable time and disk space. Documentation-only changes do not require you to
download any model.

## Make a focused change

Create a descriptive branch from the latest `main` branch:

```bash
git checkout main
git pull --ff-only
git checkout -b fix/short-description
```

Keep each pull request focused on one problem. Avoid unrelated formatting or
dependency changes. Do not commit model weights, generated images, credentials,
virtual environments, or local cache files.

## Validate your change

Run the checks relevant to your contribution. At minimum, verify that the Python
entry points compile:

```bash
python -m py_compile app.py streamlit_app.py
```

For user-interface or generation changes, also launch the Streamlit app and describe
the model, hardware, operating system, and manual checks used. Do not claim GPU or
model compatibility that you have not tested.

## Submit a pull request

Push your branch to your fork and open a pull request against `main`:

```bash
git push -u origin fix/short-description
```

In the pull request:

- Explain what changed and why.
- Link the related issue, for example `Closes #123`.
- Describe how the change was tested.
- Add screenshots for visible interface changes.
- Mention limitations, untested hardware, and model-specific behavior.
- Confirm that no secrets, model weights, or generated private content are included.

Maintainers may request changes before merging. Constructive review is part of the
contribution process.

## Reporting bugs and security concerns

Use the bug-report form for ordinary defects and include reproducible steps. Do not
open a public issue for a suspected vulnerability or exposed credential; follow
[SECURITY.md](SECURITY.md) instead.

All participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).