# Linting & Development Guide

This guide explains how to use the linting tools and the dependency management setup configured for the Eureka project.

## Dependency Management with UV

The project uses [uv](https://github.com/astral-sh/uv) as the dependency manager. Here are the common commands:

### Install All Dependencies

```bash
cd /home/daniel/media-luna/eureka
uv pip install -e .
```

### Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Add a New Dependency

```bash
uv pip install package_name
# Then add it to pyproject.toml manually
```

### Export Dependencies to requirements.txt

This is useful for Docker builds where uv might not be used:

```bash
uv pip export -o requirements.txt
```

## Linting Tools

The project is configured with several linting tools to ensure code quality:

### Running Pre-commit Hooks

Pre-commit hooks will automatically check your code on each commit. Install them once:

```bash
pre-commit install
```

To run them manually:

```bash
pre-commit run --all-files
```

### Individual Linting Tools

- **Black** (Code formatting):
  ```bash
  black .
  ```

- **isort** (Import sorting):
  ```bash
  isort .
  ```

- **Ruff** (Fast Python linter):
  ```bash
  ruff check . --fix
  ```

- **MyPy** (Type checking):
  ```bash
  mypy .
  ```

## Continuous Integration

To ensure code quality in your CI pipeline, add these steps:

```yaml
- name: Install dependencies
  run: uv pip install -e ".[dev]"

- name: Run linting
  run: |
    black --check .
    isort --check .
    ruff check .
    mypy .

- name: Run tests
  run: pytest
```

## Common Linting Issues

Here are some common issues you might encounter:

1. **Missing Type Annotations**: Add type hints to function parameters and return values
2. **Unused Imports**: Remove imports that aren't being used
3. **Line Length**: Keep lines under 88 characters
4. **Function Complexity**: Break down complex functions into smaller ones

## Best Practices

- Use consistent docstrings in the Google or NumPy format
- Follow naming conventions: snake_case for variables and functions, PascalCase for classes
- Keep functions small and focused on a single task
- Add type annotations to improve code readability and catch type errors