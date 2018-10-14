

#IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import scipy
import control
#import px4tools
#import px4tools.logsysid
import pandas
import json
from control.matlab import *
from matplotlib import *
from matplotlib import cbook

#DEN LIKER IKKE DISSE
#from mpl_toolkits.basemap import Basemap
#from matplotlib.cbook import is_scalar
#pip3 basemap

#howtoplot
import subprocess
import shlex

from scipy import integrate


#import keras #?
from keras.models import Sequential
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate, Dense, Dropout
#to_categorical ville ikke?
from keras.regularizers import l2
from pylab import rand
#from matplotlib.pylab as plt
#import network
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
#from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
#from keras_layers.keras_layer_L2Normalization import L2Normalization
#from keras_layers.keras_layer_DecodeDetections import DecodeDetections
#from keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
import tensorflow as tf

def getPidTf(kp=1,ki=0.1,kd=0):
    return TransferFunction([kd,kp,ki],[1,0])






def getError(reference, yval):
    steps = len(reference)
    if len(reference)!=len(yval):
        print("ERROR: Signal and reference need to be same dimensions")
    errorVolume = 0
    for step in range(0,steps):
        errorVolume+=abs(reference[step]-yval[step])
    return errorVolume

def sat(vin, pos_sat, neg_sat):
    if vin > pos_sat:
        out = pos_sat
    elif vin < neg_sat:
        out = neg_sat
    else:
        out = vin
    return out





def getRefsignal(start, length, mode="sine", amplitude=None, freq=None):
    ref = np.zeros(length)
    if mode == "step":
        #Creating a step input from 0 to amp
        ref[start:] = amplitude #starts at 50*resultiuon
        #print("KOM HIT")
    if mode == "sine":
        for i in range(start,length):
            ref[i]=np.sin(freq*np.pi*i/720)
            print("er her")
            #x = np.linspace(-np.pi, np.pi, 201)
            #ref[start:]=x[start:]
    if mode == "saw":
        sign = 1
        if not freq:
            freq = 500
        for i in range(start,length):
            if(i%freq):
                ref[i]=ref[i-1]+sign*amplitude/freq
            else:
                sign = sign*(-1)
                ref[i]=ref[i-1]+sign*amplitude/freq
    if mode == "square":
        sign = 1
        if not freq:
            freq = 500
        for i in range(start,length):
            if(i%freq):
                ref[i]=amplitude*sign
            else:
                sign = sign*(-1)
                ref[i]=amplitude*sign
    return ref






#Works assuming a tf (or an LTI?)

def createRefsignal(ymax,xmax,resolution=0.001):
    t = np.arange(0,xmax,resolution) #step .001, stop 5sek
    u = getRefsignal(50, len(t), mode="square", amplitude=1000)
    return u

def getResponse(ref, cltf, ymax, xmax, resolution=0.001):

    t = np.arange(0,xmax,resolution) #step .001, stop 5sek
    u = getRefsignal(50, len(t), mode="square", amplitude=1000)
    #Creating a step input from 0 to amp
    #plt.figure()
    #plt.plot(t,u)
    #plt.ylim([0,ymax])

    t, y_fb, x = control.forced_response(cltf, t, u) #outputs time, response and xout

    print("error: " + str(getError(u,y_fb)))

    print("System poles in: " + str(pole(cltf)))
    for poles in pole(cltf):
        if poles.real > 0:
            print("Closed Loop Transfer Function Unstable")
    plot = False
    if plot:
        plt.figure()
        plt.plot(t,u,t,y_fb)
        plt.show()
    return y_fb


class System:

  def __init__(sys, name, ss):
    sys.name = name
    sys.ss = ss

    #sys.TransferFunction()

  def changeName(self,newname):
    sys.name = newname
    print("Hello my name is " + sys.name)

    #print(cltf) #printer tf'en i fint format



##
#keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)


def denseModel(dim=10, layers=2, dropout=False):
  print("In fully_connected_model")
  droprate = 0.1
  model = Sequential()  # Initalize a new model
  dim=10 #for now
  model.add(Dense(units=10, activation='relu', input_dim=dim))
  #model.add(BatchNormalization())
  for i in range(layers):
      model.add(Dropout(droprate))
      model.add(Dense(units=dim, activation='relu'))
  model.add(Dense(units=3, activation='linear')) #3 cause pid, linear also probably best
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  return model


def getNeuralPidTf(kp=1,ki=0.1,kd=0):
    return TransferFunction([kd,kp,ki],[1,0])



def getLastK(signal,k):
    return signal[len(signal)-k-1:len(signal)-1] #-1pga 0index

def getInput(signal, ref):
    #test = np.append(signal,ref)
    #print("defGetpinput herer: "+str(test))

    return np.append(signal,ref)

def formatInput(signal,ref,k):
    input = getInput(getLastK(signal,k),getLastK(ref,k))
    validationRef = [getLastK(ref,k),getLastK(ref,k)]
    print("def formatInput:"+ str((input)))
    return input,validationRef
    #denne metoden vil si at NN'et har 2k inputs som den sammenligner med 2x ref

def trainANN(y_val,ref, annWidth):
    k=5
    model = denseModel(annWidth) #foreløpig dense TODO: Bytt til lstm
    inputs, validationRef = formatInput(y_val,ref,k)
    #TODO: SÅ LANGT JEG KOM, av en ellera nenn grunn er validation ref her 2xnoe. Den skal være 5xnoe.
    print("HEY BRO HERE IN TRAIN:"+str(len(validationRef)))
    print("HEY BRO HERE IN TRAIN:"+str(validationRef))

    history = model.fit(inputs, validationRef, epochs=10, batch_size=100)
    print("complete")



#print(input)


#FORELØPIG TEST TF JEG VIL STABILISERE
G = TransferFunction(30,[1,6,0])


#trenger enda et sys jeg kobler til i serie som er kontrollen altså
#closed loop TransferFunction
cltf = feedback(getPidTf(5,0.1,0.1)*G) #lager en feedback og putter alt dette i samme tf
print(cltf)

print("part 1")




ref = 1000

y_val = getResponse(1000,cltf,1050,3) #denne funksjonen plotter også
ref = createRefsignal(1000, 3)
#TODO: skill mellom referanse, amplitude osvosvself.
#feks så må jeg kødde med de 3linjene over nå bare for å gjøre begge deler
def scaleDownRef(signal,signalForScale):
    return signal[0:len(signalForScale)]
#Nå i første runde tar jeg alle punkter i hele signalet i en treningsrunde
#senere må\ jeg prpogrammere slik at alt kan skje live
ref = scaleDownRef(ref,y_val)
print("HEY BRO ref should be scaled down: "+ str(ref.shape))
trainANN(y_val,ref,10)
#model = model()  # TODO add input parameters
# TODO: Train your model (training returns history)




print("YOLO LOOK HERE:  "+ str(y_val.size))
print("yval: " + str(y_val.size))
print("ref: " + str(ref.size))
if(y_val.size != ref.size):
    print("Aint no same size")

input = np.array([y_val,ref],np.int32)
#TODO: er dette en riktig matrise?
print("input dim:"+str(input.shape))

print("Mission Accomplished")



# Plot the training using helper function created in task 0
#plot_training_score(history)

# TODO: Evaluate your model and print the score of the test set
#score = None
