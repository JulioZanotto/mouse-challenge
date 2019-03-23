import kivy
kivy.require("1.10.1")
from kivy.app import App 
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.gridlayout import GridLayout 
from kivy.clock import Clock 
import csv
import time
import datetime
import pandas as pd 
import numpy as np 
import os
from PIL import Image
from pynput import mouse
import matplotlib.pyplot as plt 

'''
def on_move(x,y):
	print ("Mouse moved to ({0}, {1})".format(x, y))

	with open('move_j1.csv','a') as csvfile:
		now = datetime.datetime.now()

		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		writer.writerow([str(now.day),",", str(now.hour),",", str(now.minute),",",str(now.second),",",x,",",y])

def on_stop():
	pass

def on_click(x, y, button, pressed):
	pass
		
def on_scroll(x, y, dx, dy):
	pass
'''
class MouseScreen(GridLayout):
	predict1 = StringProperty()
	name1 = StringProperty()


	def __init__(self,**kwargs):
		super(MouseScreen, self).__init__(**kwargs)
		Clock.schedule_interval(self.cnnPredict, 20)
		Clock.schedule_interval(self.plotImage, 10)
		
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
		#pred = model.predict(pred1)
		#self.predict1 = pred
		self.lbl.text = "Prediction goes here"
		self.lbl2.text = "Name goes here"

	def buttonClear(self):
		os.remove('im_to_pred.jpg')
		open('move_j1.csv','w').close()






class MouseTrackerApp(App):
	def build(self):
		return MouseScreen()

'''
with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
	listener.join()
'''
mouserun = MouseTrackerApp()
mouserun.run()



