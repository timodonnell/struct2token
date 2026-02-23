# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

struct2token implements adaptive all-atom tokenization of macromolecular structures (proteins, nucleic acids, small molecules). It combines two approaches:

- **All-atom tokenization** from [bio2token](https://github.com/flagshippioneering/bio2token) — retains all atoms from PDB mmCIF files
- **Adaptive tokenization** from [apt](https://github.com/rdilip/apt) — the architecture this project follows

The primary accuracy metric is all-atom RMSD after accounting for permutation symmetries.

## Current State

This is an early-stage research project. The repository currently contains only reference papers (in `references/`) and no source code implementation yet. The implementation should follow the apt architecture adapted for all-atom representation.

## Data

Training data: PDB mmCIF files located at `~/tim1/helico-data/raw/mmCIF` on the author's machine.

## Key References

- `references/apt.pdf` — Adaptive Protein Tokenization paper
- `references/bio2token.pdf` — All-atom tokenization paper
- apt source: https://github.com/rdilip/apt
- bio2token source: https://github.com/flagshippioneering/bio2token
