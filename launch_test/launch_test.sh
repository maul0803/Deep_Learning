#!/bin/bash

# Chemin vers le dossier contenant les images
DATA_FOLDER="../input_data/Data"

# Proportion des données pour l'entraînement (par exemple 70%)
TRAIN_RATIO=0.70

# Lancer le script Python avec les paramètres spécifiés pour chaque valeur de R et B

for R in 0.2 0.3 0.5 0.7
do
    for B in {10..50..10}  # De 10 à 50 avec un pas de 10
    do
        echo "Entraînement avec R=${R} et B=${B}..."
        
        # Lancer le script Python avec la valeur courante de R et B
        python3 launch_test.py --data_folder $DATA_FOLDER --train_ratio $TRAIN_RATIO --R_values $R --B_values $B
    done
done
