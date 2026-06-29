# FROST — Free-boundary Residual Operator

Source code for the paper *Learning and differentiating free-boundary equilibria for forward
simulation and inverse design* (FROST). FROST represents the solution of a free-boundary problem as
a joint fixed point of the physical field and a level set on a fixed background grid, solves and
trains that fixed point by implicit differentiation (in the manner of a deep equilibrium model), and
differentiates the same equilibrium for inverse design.

This repository ships **only the source code and the data-generation scripts** — no trained
checkpoints, predictions, figures or other results. Each script writes its own outputs to a local
`results/` folder when it is run.

## Data

All datasets used in the paper are available on Zenodo at
[10.5281/zenodo.21036347](https://doi.org/10.5281/zenodo.21036347); download each benchmark's
`*.npy` file and place it in the corresponding benchmark folder.

## Environment

Python 3.11 with PyTorch. Create the environment with

    conda env create -f environment.yml
    conda activate frost

or install the core dependencies directly:

    torch>=2.0
    numpy
    scipy
    matplotlib
    scikit-image
    shapely
    pandas

A CUDA-enabled PyTorch build is recommended for training; the scripts fall back to CPU.

## Running

Generate or download the data, train the forward operator (`C1/`), then optionally run the
inverse design (`C2_inverse/`). Per-scenario `README.md` files (in each `C1/` and `C2_inverse/`)
give the exact commands; for the obstacle problem, for example:

    python obstacle/gen_obstacle.py                   # data        -> obstacle/obstacle.npy
    python obstacle/C1/train_c1_obstacle.py           # forward op  -> obstacle/C1/results/
    python obstacle/C2_inverse/design_c2_obstacle.py  # design      -> obstacle/C2_inverse/results/

## License

Distributed under the MIT License; see `LICENSE`.

## Contact

For questions about the code please contact the corresponding author of the paper.
