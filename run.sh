#!/bin/bash

sh ./preprocess/download_abide_dataset.sh
sh ./preprocess/remove.sh
python ./preprocess/truncation.py
python ./preprocess/create_vector.py


#python create_remove.py > remove.sh                  # Create remove.sh
#wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
#python create_csv.py > abide_preprocessed.csv        # Create abide_preprocessed.csv