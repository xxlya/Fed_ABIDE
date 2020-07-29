# PyTorch implementation of Multi-site fMRI Analysis Using Privacy-preserving Federated Learning and Domain Adaptation: ABIDE Results
Our manuscript is available on arxiv https://arxiv.org/pdf/2001.05647.pdf and will be coming soon on Medical Image Analysis.

## Dependencies
- Python 3.6
- Pytorch 1.1.0
- tensorboardX

## Data
### Data Download
[ ] TBD
### Data Preprocessing
[ ] TBD

## How to run ?
Here we show a few examples using different strategies listed in the paper. Please check the meaning of configurations in each script.
### Single 
python single.py --split ${SPLIT} --site ${SITE}
### Ensemble
python ensemble.py --split ${SPLIT} --site ${SITE}
### Cross
python cross.py --trainsite ${TRAINSITE}
### MIX
python mix.py --split ${SPLIT}






