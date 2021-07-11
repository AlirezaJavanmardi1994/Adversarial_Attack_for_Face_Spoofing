import PIL
import h5py
import six
import keras
import scipy
import numpy
import yaml
!pip install git+https://github.com/rcmalli/keras-vggface.git
!pip show keras-vggface
!pip install Keras-Applications
import keras_vggface
from keras_vggface.vggface import VGGFace
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten,Dense,Dropout
from keras.applications import VGG19
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from keras_vggface import utils
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import glob
from PIL import Image
from tensorflow.keras.utils import to_categorical

if (len(x_rm_re)>len(xx_rm_re)):
  k = len(xx_rm_re)
else:
  k = len(x_rm_re)

real_labels = np.zeros((2*k,),dtype=int)
real_labels[k:2*k]=1
y_tr = real_labels

x_rm_re = x_rm_re[:k]
xx_rm_re = xx_rm_re[:k]
x_total = np.concatenate((x_rm_re,xx_rm_re))
x_total = np.array(x_total)

x_tr = x_total.astype('float32')

face_model = VGGFace(model='vgg16', include_top=False, input_shape=(176,128,3))

new_layer = face_model.output
flatten_layer = Flatten()(new_layer)
last_layer = Dropout(0.5)(flatten_layer)
last_layer = Dense(128,activation = 'relu')(last_layer)
last_layer = Dropout(0.5)(last_layer)
final_layer = Dense(2,activation = 'softmax')(last_layer)
final_model = Model(face_model.input,final_layer)
final_model.summary()

opt = Adam(lr = 1e-5)
final_model.compile(optimizer = opt , loss = 'sparse_categorical_crossentropy' , metrics=['accuracy'])
final_model.fit(x_tr,y_tr,batch_size = 128, epochs=15,shuffle=True)

tf.keras.utils.plot_model(
    final_model,
    to_file="content/model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96)

