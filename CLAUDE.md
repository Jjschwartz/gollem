# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Install: `uv sync --extra [cpu|cu121|cu124]`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/test_file.py::test_function`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run pyright`

## Code Style
- Python 3.11+ with complete type annotations
- Line length: 88 characters
- Single-line imports with 2 lines after imports section
- Use dataclasses for structured data
- Descriptive function/variable names (snake_case)
- Class names in PascalCase
- Type hints for all function parameters and return types
- Docstrings for complex functions with param descriptions
- Specific error handling with descriptive messages
- Follow existing patterns in similar files for consistency