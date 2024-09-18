# local_POD_overlap

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tobyvg.github.io/local_POD_overlap.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tobyvg.github.io/local_POD_overlap.jl/dev/)
[![Build Status](https://github.com/tobyvg/local_POD_overlap.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tobyvg/local_POD_overlap.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Abstract 

Reduced-order models (ROMs) are often used to accelerate the simulation of
large physical systems. However, traditional ROM techniques, such as those
based on proper orthogonal decomposition (POD), often struggle with advection-
dominated flows due to the slow decay of singular values. This results in high
computational costs and potential instabilities.
This paper proposes a novel approach using space-local POD to address the challenges 
arising from the slow singular value decay. Instead of global basis functions,
our method employs local basis functions that are applied across the domain,
analogous to the finite element method. By dividing the domain into subdomains
and applying a space-local POD within each subdomain, we achieve a represen-
tation that is sparse and that generalizes better outside the training regime. This
allows the use of a larger number of basis functions, without prohibitive computational costs.
To ensure smoothness across subdomain boundaries, we introduce
overlapping subdomains inspired by the partition of unity method.
Our approach is validated through simulations of the 1D advection equation
discretized using a central difference scheme. We demonstrate that using our
space-local approach we obtain a ROM that generalizes better to flow conditions
which are not part of the training data. In addition, we show that the constructed
ROM inherits the energy conservation and non-linear stability properties from
the full-order model. Finally, we find that using a space-local ROM allows for
larger time steps.

## How to run

The Experiments.ipynb file can be run using Jupyter Notebook. This file reproduces the results presented in the paper, which can be found [here](https://arxiv.org/abs/2409.08793v1).
