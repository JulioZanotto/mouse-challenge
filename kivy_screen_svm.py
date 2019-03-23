import kivy
kivy.require("1.10.0")
from kivy.app import App 
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.gridlayout import GridLayout 
from kivy.clock import Clock
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import csv
import time
import datetime
import pandas as pd 
import numpy as np 
import os
from PIL import Image
from pynput import mouse
import matplotlib.pyplot as plt
from sklearn import svm

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

X_train=np.asarray(X_train)

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))

# Train classifier and obtain predictions for OC-SVM
oc_svm = svm.OneClassSVM(gamma=0.1, kernel='rbf', nu=0.1)  # Obtained using grid search

oc_svm.fit(X_train)


class MouseScreen(GridLayout):
    predict1 = StringProperty()
    name1 = StringProperty()
    global model
    global os_svm

    def __init__(self,**kwargs):
        super(MouseScreen, self).__init__(**kwargs)
        Clock.schedule_interval(self.cnnPredict, 25)
        Clock.schedule_interval(self.plotImage, 20)

        self.predict1 = str(0)
        self.name1 = str(0)
        self.clear = 0

    def plotImage(self,dt):
        mousetrack = pd.read_csv('move_j1.csv')
        mouse_x = mousetrack.iloc[:,4].values
        mouse_y = mousetrack.iloc[:,5].values
        plt.plot(mouse_x, mouse_y, 'k')
        plt.axis('off')
        plt.savefig('im_to_pred.jpg')

    def cnnPredict(self, dt):
        im = Image.open('im_to_pred.jpg')
        pic_arr = np.asarray(im)
        pred1 = pic_arr / 255
        pred1 = np.array(pred1)
        pred1 = np.expand_dims(pred1, axis=0)
        arraylist = model.predict(pred1)
        print(arraylist.shape)
        #nsamples, nx, ny = arraylist.shape
        #arraylist = arraylist.reshape((nsamples,nx*ny))
        oc_svm_preds = oc_svm.predict(arraylist)
        if oc_svm_preds == 1:
            self.lbl.text = str(oc_svm_preds)
            self.lbl2.text = "Usuario"
        else:
            self.lbl.text = str(oc_svm_preds)
            self.lbl2.text = "Intruso!"


    def buttonClear(self):
        #os.remove('im_to_pred.jpg')
        open('move_j1.csv','w').close()

    


class MouseTrackerApp(App):
    def build(self):
        return MouseScreen()

mouserun = MouseTrackerApp()
mouserun.run()



