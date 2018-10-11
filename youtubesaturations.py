
from scipy import integrate

def mysat(vin):
    if vin > 255:
        vout = 255
    elif vin < -255:
        vout = -255
    else:
        vout = vin
    return vout

def dxdt(x,t,kp,theta_d,use_sat=True):
    theta = x[0]
    theta_dot = x[1]
    e = theta_d - theta
    v = kp*e
    if use_sat:
        v = mysat(v)
    out = [theta_dot, g*p*v-p*theta_dot]
    return out


x_mat1 = integrate.odeint(dxdt,[0,0],t,args=(5,1000,False))
x_mat2 = integrate.odeint(dxdt,[0,0],t,args=(5,1000,True))
plt.figure()
plt.plot(t,x_mat1[:,0])
plt.plot(t,x_mat2[:,0])
plt.show()

#https://github.com/ryanGT/youtube_code_share/blob/master/P_control_with_saturation/P_control_with_saturation.ipynb
# TODO: fiks denne sli kat man kan ta ulineære med saturations












#IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import scipy
import control
import px4tools
import px4tools.logsysid
import pandas
import json
from control.matlab import *
#pip3 basemap

from matplotlib import *
from matplotlib import cbook

#DEN LIKER IKKE DISSE
#from mpl_toolkits.basemap import Basemap
#from matplotlib.cbook import is_scalar

#howtoplot
import subprocess
import shlex

from scipy import integrate



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

#Works assuming a tf (or an LTI?)


def getRefsignal(start, length, mode=step, amplitude=None, freq=None):
    ref = np.zeros(length)
    if mode == "step":
        #Creating a step input from 0 to amp
        ref[start:] = amplitude #starts at 50*resultiuon
        print("KOM HIT")
    if mode == "sine":
        pass
    if mode == "saw":
        pass
    if mode == "square":
        pass
    return ref






def getResponse(ref, cltf, ymax, xmax, resolution=0.001):

    t = np.arange(0,xmax,resolution) #step .001, stop 5sek


    u=getRefsignal(50,len(t),"step")
    #plt.figure()
    #plt.plot(t,u)
    plt.ylim([0,ymax])

    t, y_fb, x = control.forced_response(cltf, t, u) #outputs time, response and xout

    print("error: " + str(getError(u,y_fb)))

    print("System poles in: " + str(pole(cltf)))
    for poles in pole(cltf):
        if poles.real > 0:
            print("Closed Loop Transfer Function Unstable")
    plt.figure()
    plt.plot(t,u,t,y_fb)
    plt.show()
