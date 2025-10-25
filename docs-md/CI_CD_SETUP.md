# CI/CD Setup Guide

## Overview

This repository uses GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD). All workflows are configured in `.github/workflows/`.

## Workflows

### 1. Tests (`tests.yml`)

**Trigger**: Push or PR to `main` or `develop` branches

**Purpose**: Run the test suite across multiple Python versions and operating systems

**Matrix Testing**:
- **Operating Systems**: Ubuntu, macOS, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Total combinations**: 15 test runs per trigger

**Steps**:
1. Checkout code
2. Set up Python environment with caching
3. Install dependencies (including dev dependencies)
4. Run pytest with coverage
5. Upload coverage to Codecov (Ubuntu + Python 3.11 only)

**Badge**: 
```markdown
[![Tests](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/tests.yml)
```

### 2. Lint (`lint.yml`)

**Trigger**: Push or PR to `main` or `develop` branches

**Purpose**: Check code quality and formatting

**Checks**:
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting and style guide enforcement

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install linting tools
4. Run black check (fails if code needs formatting)
5. Run isort check (fails if imports need sorting)
6. Run flake8 (fails on style violations)

**Badge**: 
```markdown
[![Lint](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/lint.yml)
```

**Note**: Run `make format` locally before pushing to avoid failures.

### 3. Documentation (`docs.yml`)

**Trigger**: Push or PR to `main` or `develop` branches

**Purpose**: Build Sphinx documentation to ensure it compiles without errors

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install dependencies (including Sphinx)
4. Build HTML documentation
5. Upload built documentation as artifact (retained for 7 days)

**Badge**: 
```markdown
[![Documentation](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/docs.yml/badge.svg)](https://github.com/AndreasTersenov/wl_stats_torch/actions/workflows/docs.yml)
```

**Accessing Built Docs**: Download from workflow run artifacts in GitHub Actions tab.

### 4. Publish to PyPI (`publish.yml`)

**Trigger**: 
- Automatic: When a GitHub release is published
- Manual: Workflow dispatch (can publish to TestPyPI)

**Purpose**: Build and publish package to PyPI

**Jobs**:

#### Job 1: Build Distribution
- Builds source distribution (`.tar.gz`)
- Builds wheel distribution (`.whl`)
- Validates packages with `twine check`
- Uploads artifacts

#### Job 2: Publish to PyPI (automatic)
- Runs only on release publication
- Downloads build artifacts
- Publishes to PyPI using trusted publishing (no API tokens needed)
- Requires: PyPI trusted publisher configured

#### Job 3: Publish to TestPyPI (manual)
- Runs only on manual workflow dispatch with `test_pypi=true`
- Downloads build artifacts
- Publishes to TestPyPI for testing
- Requires: TestPyPI trusted publisher configured

**Security**: Uses OpenID Connect (OIDC) for authentication (no secrets needed).

## Setup Instructions

### 1. Enable Workflows

Workflows are automatically enabled when pushed to GitHub. No additional setup needed.

### 2. Configure Branch Protection (Optional but Recommended)

Go to: `Settings` → `Branches` → Add rule for `main`

Recommended settings:
- ✅ Require status checks to pass before merging
  - Required checks: `Test Python 3.11 on ubuntu-latest`, `Code Quality Checks`, `Build Documentation`
- ✅ Require branches to be up to date before merging
- ✅ Require linear history

### 3. Set Up PyPI Trusted Publishing

#### For PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new trusted publisher:
   - **PyPI Project Name**: `wl-stats-torch`
   - **Owner**: `AndreasTersenov`
   - **Repository name**: `wl_stats_torch`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

#### For TestPyPI (optional, for testing):

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new trusted publisher with same settings but:
   - **Environment name**: `testpypi`

### 4. Set Up Codecov (Optional)

1. Go to https://codecov.io/
2. Sign in with GitHub
3. Add repository: `AndreasTersenov/wl_stats_torch`
4. No token needed for public repositories

### 5. Configure GitHub Environments (for PyPI publishing)

Go to: `Settings` → `Environments`

Create two environments:

#### Environment: `pypi`
- **Deployment protection rules** (optional):
  - ✅ Required reviewers: Add yourself or team members
  - ⏱️ Wait timer: 5 minutes (prevents accidental deploys)

#### Environment: `testpypi`
- No special protection needed (for testing only)

## Usage

### Running Tests Locally

Before pushing, run the same checks locally:

```bash
# Run all checks
make all

# Individual checks
make test      # Run tests
make lint      # Check linting
make format    # Auto-format code
make docs      # Build documentation
```

### Creating a Release

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. **Update changelog** (if you have one):
   ```bash
   # Add to CHANGELOG.md or docs-md/CHANGELOG.md
   ```

3. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push
   ```

4. **Create GitHub release**:
   - Go to: https://github.com/AndreasTersenov/wl_stats_torch/releases/new
   - Tag: `v0.2.0`
   - Title: `Release 0.2.0`
   - Description: List changes and improvements
   - Click "Publish release"

5. **Automatic deployment**:
   - GitHub Actions automatically builds and publishes to PyPI
   - Watch the workflow: Actions → Publish to PyPI

### Testing PyPI Upload (Optional)

To test publishing without affecting the real PyPI:

1. Go to: Actions → Publish to PyPI
2. Click "Run workflow"
3. Check "Publish to Test PyPI instead of PyPI"
4. Run workflow

Then install from TestPyPI to verify:
```bash
pip install --index-url https://test.pypi.org/simple/ wl-stats-torch
```

## Workflow Status

Check workflow status:
- **GitHub Actions Tab**: https://github.com/AndreasTersenov/wl_stats_torch/actions
- **README Badges**: Show current status at a glance

## Troubleshooting

### Tests Failing

**Symptoms**: Red ❌ badge, failed workflow runs

**Solutions**:
1. Check workflow logs in GitHub Actions
2. Run tests locally: `make test`
3. Ensure all dependencies are correctly specified in `pyproject.toml`
4. Check if test works on your local OS but fails on others (OS-specific issue)

### Linting Failing

**Symptoms**: Lint workflow fails

**Solutions**:
1. Run locally: `make format && make lint`
2. Fix any violations
3. Commit and push fixes

### Documentation Build Failing

**Symptoms**: Docs workflow fails

**Solutions**:
1. Run locally: `make docs`
2. Check Sphinx errors
3. Fix documentation syntax issues
4. Ensure all referenced files exist

### PyPI Publishing Failing

**Symptoms**: Publish workflow fails

**Common issues**:
1. **Version already exists**: Bump version in `pyproject.toml`
2. **Trusted publisher not configured**: Follow setup instructions above
3. **Package name taken**: Choose different name in `pyproject.toml`

**Debug**:
1. Check workflow logs
2. Test with TestPyPI first
3. Verify `pyproject.toml` metadata is correct

## Best Practices

### Before Every Push

```bash
# Format code
make format

# Check everything
make all

# Only push if all checks pass
git push
```

### Before Creating Release

```bash
# Ensure everything is working
make all

# Build and check package
make build
ls -lh dist/

# Verify version is correct
grep version pyproject.toml
```

### Semantic Versioning

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Example: `0.1.0` → `0.2.0` (new features) → `0.2.1` (bug fix) → `1.0.0` (stable API)

## Additional Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **PyPI Publishing**: https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- **Trusted Publishers**: https://docs.pypi.org/trusted-publishers/
- **Codecov Docs**: https://docs.codecov.com/

## Workflow Files

All workflow files are in `.github/workflows/`:
- `tests.yml` - Test suite
- `lint.yml` - Code quality
- `docs.yml` - Documentation build
- `publish.yml` - PyPI publishing

Feel free to customize these workflows to fit your needs!
