# state-space-estimation

## Installation

This project uses `pyproject.toml` for modern Python packaging. Below are the common installation methods:

### Quick Reference

| Use Case | Command |
|----------|---------|
| **Basic install** | `pip install .` |
| **Editable (dev)** | `pip install -e .` |
| **With dev tools** | `pip install -e ".[dev]"` |
| **With kinematics** | `pip install -e ".[kinematics]"` |
| **Everything** | `pip install -e ".[dev,kinematics,docs]"` |
| **From GitHub** | `pip install git+https://github.com/yourusername/state-space-estimation.git` |
| **Upgrade** | `pip install --upgrade state-space-estimation` |
| **Uninstall** | `pip uninstall state-space-estimation` |

### Installation Modes Explained

**Basic Install** (`pip install .`)
- Installs the package in standard mode
- Good for testing before publishing
- Changes to code require reinstallation

**Editable Mode** (`pip install -e .`)
- Installs in development mode
- Changes to your code are immediately reflected
- No need to reinstall after editing
- Recommended for active development

**With Optional Dependencies**
- `[dev]` - Development tools (pytest, black, flake8, mypy, isort)
- `[kinematics]` - Kinematics support (numpy-quaternion)
- `[docs]` - Documentation tools (sphinx, sphinx-rtd-theme)
- Combine multiple extras: `pip install -e ".[dev,kinematics]"`

### Development Setup

For contributing to this project:

```bash
# Clone the repository
git clone https://github.com/yourusername/state-space-estimation.git
cd state-space-estimation

# Install in editable mode with all development tools
pip install -e ".[dev,kinematics,docs]"
```

### Project Structure

The project is organized as follows:
- `bayesian_framework/` - Core Bayesian filtering framework
- `kinematics/` - Kinematics and motion models
- `sensors/` - Sensor models (IMU, etc.)
- `state_space_models/` - State-space model implementations
- `utils/` - Utility functions for matrices, plotting, statistics
