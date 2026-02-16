# Project Structure Conventions

## Directory Organization

### src/
- Contains reusable utility modules
- Flat structure (no nested packages)

### scripts/
- Contains runnable scripts
- Each script in its own subdirectory: `scripts/<script_name>/`
- Each subdirectory contains:
  - `run.py` - Entry point (execute via `python scripts/<script_name>/run.py`)
  - `config.yaml` - Configuration (auto-loaded from same directory)

### tests/
- Mirrors the structure of both `src/` and `scripts/`
- `tests/src/` - Tests for modules in `src/`
- `tests/scripts/` - Tests for scripts in `scripts/`
- Test file naming: `test_<module_name>.py`

## Configuration
- All user inputs via YAML files
- YAML files co-located with scripts that use them
