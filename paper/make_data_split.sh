#!/usr/bin/env bash

#conda env create -f ../environment.yml

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate immunet_dev

python data_split.py --stage bladder
python data_split.py --stage lung
python data_split.py --stage melanoma
python data_split.py --stage prostate
python data_split.py --stage tonsils
python data_split.py --stage makeSplit
python data_split.py --stage saveStat
