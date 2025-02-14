# Librairies standards
import os

# Librairies scientifiques et de traitement d'image
import numpy as np
import cv2

# Librairies de machine learning
from tensorflow import keras
import tensorflow as tf

# Versions de données
from numpy import uint8

def PSNR(x_true, x_pred):
    x_true = tf.cast(x_true, tf.float32)
    max_I = 255.0
    mse = tf.reduce_mean(tf.square(x_pred - x_true))
    psnr_value = tf.where(tf.equal(mse, 0), 100.0, 10.0 * tf.math.log((max_I ** 2) / mse) / tf.math.log(10.0))

    return tf.cast(psnr_value, tf.float32)

    return psnr_value
def phi(M,N):
    p = 1/np.sqrt(M)
    res = np.zeros((M,N))
    res =  np.random.normal(0,p,M*N).reshape(M,N)
    #res = np.random.uniform(-p,p,M*N).reshape(M,N)
    return(res)

# images = la liste de toutes les images de l'entrainement, ici on a 100 images
def input_images_to_vectors(images,B):
    input_X = []
    for i in images:
        input_X.extend(image_to_vectors(i,B))
    return np.asarray(input_X, dtype=uint8)

#liste = liste des vecteurs colonnes (les blocs vectorisés et mis en array)
#img_shape = la taille de l'image attendue
def vectors_to_image(liste,B,img_shape):
    numBlocks_R = int(img_shape[0]/B)
    numBlocks_C = int(img_shape[1]/B)
    newImage=np.empty((img_shape[0],img_shape[1]))
    for r in range(numBlocks_R):
        for c in range(numBlocks_C):
            if liste.size != 0:
                tmp = liste[0].reshape((B,B))
                liste = np.delete(liste,0,0)
                newImage[r*B:(r+1)*B , c*B:(c+1)*B] = tmp

    return newImage

#B : la taille du bloc BxB
def image_to_vectors(image,B):

    R,C = image.shape # shape de l'image
    res = [] # liste de vectors
    for r in range(0,R-B+1,B): # Parcours de l'image par blocs de taille BxB
        for c in range(0,C-B+1,B):
            tmp = image[r:B+r,c:c+B].flatten() # Extraction d'un bloc et aplatissement en vecteur
            res.append(tmp) # Ajout du vecteur dans la list

    return np.asarray(res)

def images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        imagePath = os.path.join(folder,filename)
        img = cv2.imread(imagePath,cv2.IMREAD_UNCHANGED)
        images.append(img[:,:,0])
    return images

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.PSNR = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.PSNR.append(logs.get('PSNR'))