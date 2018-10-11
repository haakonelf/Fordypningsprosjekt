




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






#TEST
def test():
    print("test")
    for i in range(0,3):
        print(i)

test()
print("enda en test")



#sys = StateSpace(A,B,C,D)
#tf = TransferFunction(num,den)


class System:

  def __init__(sys, name, ss):
    sys.name = name
    sys.ss = ss

    #sys.TransferFunction()

  def changeName(self,newname):
    sys.name = newname
    print("Hello my name is " + sys.name)


A=1
B=1
C=1
D=1
p1 = System("John", StateSpace(A,B,C,D))
p1.changeName('bitchassniggu')



#EKSEMPEL FRA YOUTUGBE


#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import control



p = 6
g = 5
G = control.TransferFunction(g*p,[1,p,0])

kp = 5
cltf = control.feedback(kp*G)
print(cltf)

t = np.arange(0,3,0.001)
amp = 1000
u = np.zeros(len(t))
u[50:] = amp
plt.figure()
plt.plot(t,u)
plt.show()

plt.ylim([0,1050])
t, y_fb, x = control.forced_response(cltf, t, u)
y_fb.shape
x.shape
plt.figure()
plt.plot(t,u,t,y_fb)
plt.show()


t, y_fb, x = control.forced_response(cltf, t, u)


#SIMULATION
n = len(G.pole())
x_prev = np.zeros(n)
y_one_step = np.zeros(len(t))
dt = t[1]-t[0]
pwm_vect = np.zeros(len(t))

import pdb

for i, t_i in enumerate(t):
    e = u[i] - y_one_step[i-1]
    pwm = kp*e
    pwm_vect[i] = pwm
    t_temp, y_temp, x_temp = control.forced_response(G,[t_i-dt,t_i],[pwm,pwm], X0=x_prev)
    y_one_step[i] = np.squeeze(y_temp[-1])
    x_prev = np.squeeze(x_temp[:,-1])

x_temp
y_one_step
print(plt.figure())
plt.plot(t,u,t,y_fb,'y')
plt.plot(t, y_one_step,'k:',linewidth=3)
plt.show()

plt.figure()
plt.plot(t,pwm_vect)
G.pole()
plt.show()








n=20
x=np.linspace(0,np.pi,n)
y=np.sin(x)
plt.plot(x,y)
fname='/tmp/test.pdf'
plt.savefig(fname)
proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
plt.show()
