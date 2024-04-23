# Coral Microatoll Simulator

This repository contains python code for the forward simulation of coral microatoll growth based on several parameters.

## Installation

To install, first create a new environment(recommended):

```bash
conda create --name microatoll_sim
conda activate microatoll_sim
```

Install the project's dependencies and the project library itself:

```bash
python -m pip install . # will install core dependencies via pyproject.toml
python -m pip install .\[all\] # will install dependencies needed to run examples in the examples/ folder
```

You can now import it in a python script as follows:

```python
import microatoll_sim.simulator as sim
sim.coral_growth(...)
```

For a simple example as well as function benchmarks, check out `examples/example.ipynb`.
