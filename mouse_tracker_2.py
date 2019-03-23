from pynput.mouse import Listener
import csv
import datetime
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

count = 0

with open('move_j1.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['day',',','hour',',','minute',',','second',',','x',',','y'])

'''
Mouse events handler
________________________________________________________________________________________
'''

def on_move(x,y):
	print ("Mouse moved to ({0}, {1})".format(x, y))

	with open('move_j1.csv','a') as csvfile:
		now = datetime.datetime.now()

		writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

		writer.writerow([str(now.day),",", str(now.hour),",", str(now.minute),",",str(now.second),",",x,",",y])

def on_stop():
	print("stop")

def on_click(x, y, button, pressed):
	if pressed:
		print("Mouse click at ({0}, {1}) with {2}".format (x, y, button))
		
def on_scroll(x, y, dx, dy):
	print(x, y, dx, dy)
	print("Mouse scrolled at ({0}, {1})({2}, {3})".format(x, y, dx, dy))

'''
________________________________________________________________________________________
'''

'''
def save_file():
        now = datetime.datetime.now()
        count = now.second
        f = open('out.txt','a')
        f.write("thread test"+str(count)+'\n')
        f.close()
        threading.Timer(10, save_file).start()
'''


def image_creator():
        now = datetime.datetime.now()
        minute = now.minute
        sec = now.second
        count = str(minute)+str(sec)
        mousetrack = pd.read_csv('move_j1.csv')
        mouse_x = mousetrack.iloc[:,4].values
        mouse_y = mousetrack.iloc[:,5].values
        plt.plot(mouse_x, mouse_y, 'k')
        plt.axis('off')
        plt.savefig('./train/julio/fig'+str(count)+'.jpg')
        threading.Timer(5, image_creator).start()

#/Users/Julio/Downloads/I2A2/fig
        
image_creator()

with Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
	listener.join()


#save_file()

#listener = Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)

#listener.start()

                
                
        




