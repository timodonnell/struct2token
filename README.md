# struct2token
Adaptive all-atom tokenization of arbitrary proteins, nucleic acids, and small molecules

This repository implements an all-atom *adaptive* tokenization scheme for macromolecular structures.

It combines two existing ideas:

**All atom tokenization** as implemented in [bio2token](https://github.com/flagshippioneering/bio2token).

**Adaptive tokenization** as implemented in [apt](https://github.com/rdilip/apt).

Basically, we wanted apt, but for all atoms, so we implemented this. We follow the apt architecture.

We train on pdb mmcifs, which on my machine are located at. We retain all atoms in the mmcif. Our primary accuracy metric is all atom rmsd, after accounting for permutation symmetries.

## Data
Specific to my machine. We have pdb mmcifs at: ~/tim1/helico-data/raw/mmCIF
