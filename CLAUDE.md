# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Stanford CS336 Spring 2025 Assignment 1: Basics - implementing transformer model components and BPE tokenizers from scratch in PyTorch. Students implement functions in `tests/adapters.py` that are tested against reference implementations.

## Development Environment

- **Package Manager**: `uv` (modern Python package manager)
- **Python**: ≥3.11
- **PyTorch**: ~2.6.0 (special handling for Intel Macs: ~2.2.2)
- **Code Quality**: `ruff` with 120-character line length
- **Type Safety**: `jaxtyping` for tensor type annotations

## Common Commands

### Environment Management
```bash
# Run any Python file with uv-managed environment
uv run <python_file_path>

# The environment is automatically solved and activated when needed
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_model.py

# Run specific test function
uv run pytest tests/test_model.py::test_linear

# Run tests with verbose output
uv run pytest -v

# Create submission package (runs tests first)
./make_submission.sh
```

### Code Quality
```bash
# Lint with ruff (configured in pyproject.toml)
uv run ruff check .

# Format with ruff
uv run ruff format .
```

## Architecture & Implementation Patterns

### Adapter Pattern
- Student implementations go in `tests/adapters.py`
- Each function in `adapters.py` corresponds to test cases
- Functions receive weights/tensors and must return correct outputs
- Reference implementations use snapshot testing (`_snapshots/` directory)

### Test Structure
- **`test_data.py`**: Data loading and batch sampling (`run_get_batch`)
- **`test_model.py`**: Core transformer components (Linear, Embedding, SwiGLU, Attention, RMSNorm, RoPE, Transformer blocks)
- **`test_tokenizer.py`**: BPE tokenizer implementation (`get_tokenizer`)
- **`test_train_bpe.py`**: BPE training algorithm (`run_train_bpe`)
- **`test_optimizer.py`**: Optimizer implementations (AdamW, learning rate schedules)
- **`test_nn_utils.py`**: Neural network utilities (softmax, cross-entropy, gradient clipping)
- **`test_serialization.py`**: Model checkpoint saving/loading

### Key Implementation Areas
1. **Transformer Components**:
   - Linear layers, Embedding layers
   - RMSNorm, SwiGLU, RoPE (Rotary Positional Encoding)
   - Multi-head self-attention (with and without RoPE)
   - Scaled dot-product attention
   - Full transformer blocks and language models

2. **Tokenizer**:
   - BPE (Byte Pair Encoding) tokenizer
   - Tokenizer training from text corpus

3. **Training Infrastructure**:
   - AdamW optimizer
   - Cosine learning rate schedule with warmup
   - Gradient clipping
   - Model serialization

### Data Handling
- Datasets: TinyStoriesV2 and OpenWebText sample
- Data directory: `data/` (created by download script in README)
- `run_get_batch` samples sequences for language modeling

### Type Annotations
- Uses `jaxtyping` for tensor shape annotations
- Example: `Float[Tensor, "batch sequence_length d_model"]`
- Functions in `adapters.py` have detailed type hints

## Workflow for Students

1. **Setup**: Install `uv`, run `uv run pytest` (all tests fail initially)
2. **Implement**: Fill in functions in `tests/adapters.py`
3. **Test**: Run `uv run pytest` to check implementations
4. **Submit**: Run `./make_submission.sh` to create submission zip

## Important Notes

- **Snapshot Testing**: Expected outputs stored in `tests/_snapshots/`
- **Platform-specific Dependencies**: PyTorch version differs for Intel Macs
- **Submission**: `make_submission.sh` excludes large files (datasets, checkpoints, etc.)
- **Dataset Download**: See README for wget commands to download TinyStories and OpenWebText

## File Organization

```
cs336_basics/          # Main package (minimal - mostly educational)
tests/                 # Test suite and student implementations
├── adapters.py        # STUDENT IMPLEMENTATIONS GO HERE
├── common.py          # Shared test utilities
├── conftest.py        # Pytest configuration
├── fixtures/          # Test data
├── _snapshots/        # Expected outputs for snapshot tests
└── test_*.py          # Test modules by category
data/                  # Dataset storage (created by user)
datasets/              # Example dataset included
```

## When Helping with Implementation

- Focus on completing functions in `tests/adapters.py`
- Use type hints and shape annotations from function signatures
- Reference the assignment PDF for detailed requirements
- Test frequently with `uv run pytest`
- Check snapshot outputs if tests fail unexpectedly