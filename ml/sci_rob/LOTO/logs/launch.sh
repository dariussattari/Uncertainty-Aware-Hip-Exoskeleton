#!/bin/bash
# GAN only. The four LOSO-based ensembles are deferred to the HPC cluster -- see README.
cd /Users/dariussattari/Documents/thesis/Uncertainty-Aware-Hip-Exoskeleton
exec .venv/bin/python -u ml/sci_rob/LOTO/run.py --all --model vanilla/GAN
