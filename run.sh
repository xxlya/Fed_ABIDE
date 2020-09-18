#!/bin/bash

<<<<<<< HEAD
sh ./preprocess/download_abide_dataset.sh
sh ./preprocess/remove.sh
python ./preprocess/truncation.py
python ./preprocess/create_vector.py
=======
cd preprocess
sh download_abide_dataset.sh
sh remove.sh
python truncation.py
python create_vector.py
>>>>>>> 720ce4e0a4fca75dc23a918c3b6a1633e002a865


#python create_remove.py > remove.sh                  # Create remove.sh
#wget https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
<<<<<<< HEAD
#python create_csv.py > abide_preprocessed.csv        # Create abide_preprocessed.csv
=======
#python create_csv.py > abide_preprocessed.csv        # Create abide_preprocessed.csv
>>>>>>> 720ce4e0a4fca75dc23a918c3b6a1633e002a865
