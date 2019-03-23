from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import numpy as np
import os
from PIL import Image

def model_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(480, 640,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    return model

model = model_cnn()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def extract_flatten(X):
    global model
    features_array = model.predict(X)
    return features_array

X_train=[]
for file in os.listdir('./train/usuario/'):
     im = Image.open('./train/usuario/'+str(file))
     pic_arr = np.asarray(im)
     pred1 = pic_arr / 255
     pred1 = np.array(pred1)
     pred1 = np.expand_dims(pred1, axis=0)
     arraylist = extract_flatten(pred1)
     X_train.append(arraylist)



X_test=[]
for file in os.listdir('./train/intruso/'):
     im = Image.open('./train/intruso/'+str(file))
     pic_arr = np.asarray(im)
     pred1 = pic_arr / 255
     pred1 = np.array(pred1)
     pred1 = np.expand_dims(pred1, axis=0)
     arraylist = extract_flatten(pred1)
     X_test.append(arraylist)

X_train=np.asarray(X_train)
X_test=np.asarray(X_test)

from sklearn import svm

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

# Train classifier and obtain predictions for OC-SVM
oc_svm_clf = svm.OneClassSVM(gamma=0.1, kernel='rbf', nu=0.1)  # Obtained using grid search

oc_svm_clf.fit(X_train)

oc_svm_preds = oc_svm_clf.predict(X_test)

print(oc_svm_preds)

