# Librairies de machine learning
from tensorflow import keras
from keras.models import load_model
from keras.utils import custom_object_scope
from useful import LossHistory, images_from_folder, image_to_vectors, vectors_to_image, input_images_to_vectors, phi, PSNR
import tensorflow as tf
from tensorflow.python.framework import ops
from model import build_Model
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K #numpy() is only available when eager execution is enabled.
import argparse
# Librairies standards
import os
import random
import time
import json
from tensorflow.keras.callbacks import EarlyStopping  # 1. Importer EarlyStopping

def clear_gpu_cache():
    tf.keras.backend.clear_session()
    
    ops.reset_default_graph()
    print("Mémoire GPU libérée.")

def train_and_save_model(B, R, y_train, y_test, X_train, X_test, RMSprop_optimizer, T=8, model_path="trained_model.keras"):
    """
    Entraîne et sauvegarde un modèle de reconstruction d'images basé sur l'apprentissage profond avec early stopping.
    
    Paramètres :
    - B : Taille des blocs (BxB)
    - X_train : Données d'entraînement
    - X_test : Données de test
    - T : Paramètre du modèle (par défaut 8)
    - model_path : Chemin où sauvegarder le modèle entraîné
    
    Retourne :
    - L'historique d'entraînement et le modèle entraîné
    """ 
    
    model = build_Model(B, T, R)
    model.compile(optimizer=RMSprop_optimizer, loss='mean_squared_error', metrics=[PSNR])

    print("Dimensions des données :")
    print(f"y_train: {np.shape(y_train)}, X_train: {np.shape(X_train)}")
    print(f"y_test: {np.shape(y_test)}, X_test: {np.shape(X_test)}")

    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)
    X_test = tf.convert_to_tensor(X_test)
    y_test = tf.convert_to_tensor(y_test)
    loss_history = LossHistory()

    # 2. Définir l'early stopping
    early_stopping = EarlyStopping(monitor='val_loss',  # Critère : val_loss (ou 'val_PSNR')
                                   patience=5,          # Nombre d'époques sans amélioration avant l'arrêt
                                   verbose=1,          # Afficher des messages lors de l'arrêt anticipé
                                   restore_best_weights=True)  # Restaurer les meilleurs poids du modèle

    start = time.time()
    history = model.fit(
        y_train, X_train,
        batch_size=64, epochs=50,  # EPOCH
        verbose=1,
        validation_data=(y_test, X_test),
        callbacks=[loss_history, early_stopping] if 'loss_history' in globals() else [early_stopping]  # 3. Ajouter le callback
    )
    end = time.time()

    print(f"Temps d'entraînement : {end - start:.2f} s")
    model.save(model_path)
    print(f"Modèle sauvegardé sous {model_path}")

    return history, model


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur l'ensemble de test.
    """
    X_test = tf.convert_to_tensor(X_test)
    y_test = tf.convert_to_tensor(y_test)

    loss, psnr = model.evaluate(y_test, X_test, verbose=1)
    print(f"Perte (Loss) : {loss:.4f}")
    print(f"PSNR : {psnr:.4f} dB")
    return loss, psnr

def load_model_if_exists(model_path, optimizer):
    """
    Charge le modèle s'il existe, sinon retourne None.
    """
    if False: #os.path.exists(model_path):
        print(f"Chargement du modèle existant : {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[PSNR])
        return model
    return None
def launch_test(list_R, list_B, XX_train, XX_test, T=8):
    """
    Entraîne plusieurs modèles si nécessaire et sauvegarde leurs résultats.
    """
    results_path = "results/all_results.json"
    models_dir = "models"
    os.makedirs("results", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists(results_path):
        with open(results_path, "r") as json_file:
            try:
                all_results = json.load(json_file)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    existing_configs = {(res["R"], res["B"]) for res in all_results}

    RMSprop_optimizer = keras.optimizers.RMSprop(
        learning_rate=0.0001,
        rho=0.9,
        momentum=0.0,
        weight_decay=1e-7
    )

    for B in list_B:
        X_train = input_images_to_vectors(XX_train, B)
        X_test = input_images_to_vectors(XX_test, B)

        for R in list_R:
            model_path = f"{models_dir}/R_{R}_size_patch_{B}_trained_model.keras"

            if (R, B) in existing_configs:
                print(f"Résultats déjà enregistrés pour R={R}, B={B}, saut du recalcul.")
                continue

            model = load_model_if_exists(model_path, RMSprop_optimizer)

            M = int(np.floor(B**2 * R))
            N = B**2
            Phi = phi(M, N)

            y_train = (Phi @ X_train.T).T
            y_test = (Phi @ X_test.T).T

            if model is None:
                print(f"Entraînement du modèle pour R={R}, B={B}")
                history, model = train_and_save_model(B, R, y_train, y_test, X_train, X_test, RMSprop_optimizer, T, model_path)
                loss_key, PSNR_key, val_loss_key, val_PSNR_key = history.history.keys()
                result = {
                    "R": R,
                    "B": B,
                    "loss": history.history[loss_key][-1],
                    "PSNR": history.history[PSNR_key][-1],
                    "val_loss": history.history[val_loss_key][-1],
                    "val_PSNR": history.history[val_PSNR_key][-1],
                }
            else:
                print(f"Évaluation du modèle existant pour R={R}, B={B}")
                loss, psnr = evaluate_model(model, X_train, y_train)
                val_loss, val_psnr = evaluate_model(model, X_test, y_test)

                result = {
                    "R": R,
                    "B": B,
                    "loss": loss,
                    "PSNR": psnr,
                    "val_loss": val_loss,
                    "val_PSNR": val_psnr,
                }

            all_results.append(result)
            #clear_gpu_cache()
    # Sauvegarder tous les résultats dans un fichier JSON
    with open(results_path, "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"Tous les résultats mis à jour sous {results_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement du modèle avec des paramètres d'entrée.")
    
    parser.add_argument('--data_folder', type=str, required=True, help="Chemin vers le dossier contenant les images.")
    parser.add_argument('--train_ratio', type=float, default=0.70, help="Proportion des données utilisées pour l'entraînement.")
    parser.add_argument('--R_values', type=float, nargs='+', default=[0.7], help="Liste des valeurs de R à tester.")
    parser.add_argument('--B_values', type=int, nargs='+', default=[50], help="Liste des tailles de blocs B à tester.")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parser les arguments
    args = parse_args()
    # Charger les images depuis le dossier spécifié
    images = images_from_folder(args.data_folder)
    # Séparer le dataset en 2 : 70% pour l'entraînement et 30% pour le test
    XX_train, XX_test, yy_train, yy_test = train_test_split(images, images, test_size=1 - args.train_ratio, random_state=42)
    # Lancer le test avec les paramètres donnés
    launch_test(args.R_values, args.B_values, XX_train, XX_test)
