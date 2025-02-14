# Librairies scientifiques et de traitement d'image
import numpy as np

# Librairies de machine learning
import tensorflow as tf
from tensorflow import keras

def build_Model(B, T, R):
    """
    Construit un modèle de reconstruction basé sur un réseau de neurones.
    
    Paramètres :
    - B : Taille des blocs (BxB)
    - T : Profondeur du modèle
    - R : Ratio de mesure (R = M / B²)
    
    Retourne :
    - Un modèle Keras compilé
    """
    ## Définition du nombre d'entrées et sorties
    M = int(np.floor(B * B * R))  # Nombre de mesures après projection
    input_dim = M  # Dimension d'entrée
    output_dim = B * B  # Dimension de sortie

    ## Set random seeds pour reproductibilité
    np.random.seed(23)
    tf.random.set_seed(23)

    # Création du modèle séquentiel
    model = keras.models.Sequential([
        # Première couche cachée
        keras.layers.Dense(
            units=B * B * T,
            input_dim=input_dim,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'
        ),
        
        # Deuxième couche cachée
        keras.layers.Dense(
            units=B * B * T,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu'
        ),
        
        # Couche de sortie (pas d'activation pour garder une sortie continue)
        keras.layers.Dense(
            units=output_dim,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
    ])

    print(f"✅ Modèle construit avec input_dim={input_dim}, output_dim={output_dim}")
    
    return model
