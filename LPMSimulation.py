from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
import time


@njit
def Single_run_numba(Tmax,rn,rf,Dmax,Dmin,A1,b):
############################## INITIALIZATION ############################################
    RT = 0
    click = 0
    X = np.ones(3)*0.001 
    acc = 0
    while (acc == 0):
        X[0] = rn - 2*rn*np.random.rand()
        X[1] = rn - 2*rn*np.random.rand()
        X[2] = rn - 2*rn*np.random.rand()
        r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2);
        if (r < rn):
             acc = 1

    dX = np.zeros(3) 
    rxp = np.zeros(3) 
    dXpX = np.zeros(3) 
    a2r = np.zeros(3) 
    r1new = np.zeros(3) 
    rnew = np.zeros(3) 
    For = np.zeros(3) 
    cc = 0.0
    r01 = 0.0    
    SIRfile = []
    BB = 0
    hf = 1.0
    num = 0;
    count = 0
    nts = 0.02

############################## RUN SCRIPT    ############################################
    while(RT < Tmax):
        RT += dt;
        num += 1;

        r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2);
        gx = 1. + np.exp(-b*(r-rf));
        DDX = Dmin + (Dmax-Dmin)/gx;
        Dterm = np.sqrt(2*DDX*dt);
        dDiff = (Dmax-Dmin)*b*np.exp(-b*(r-rf))/(gx*gx);
        Pote = dt/r*(DDX*A1 - (Dmax-Dmax))*b*np.exp(-b*(r-rf))/(gx*gx)
#        Pote = 0; DDX = 1.0;
        dX[0] = -X[0]*Pote + np.sqrt(2*DDX*dt)*np.random.normal()
        dX[1] = -X[1]*Pote + np.sqrt(2*DDX*dt)*np.random.normal()
        dX[2] = -X[2]*Pote + np.sqrt(2*DDX*dt)*np.random.normal()
        r0l = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
        dXpX[0] = X[0] + dX[0];  
        dXpX[1] = X[1] + dX[1];  
        dXpX[2] = X[2] + dX[2];                                                                                                                 

######### Reflect ############
        rw = np.sqrt(dXpX[0]**2 + dXpX[1]**2 + dXpX[2]**2)
        if (rw > rn):
            cc = (rn-r0l)/(rw-r0l);                                                                                                                                  
            rxp[0] = X[0] + cc*dX[0];  rxp[1] = X[1] + cc*dX[1];  rxp[2] = X[2] + cc*dX[2];                                                                   
            dotp = dX[0]*rxp[0]/rn + dX[1]*rxp[1]/rn + dX[2]*rxp[2]/rn;                                                                                
            a2r[0] = dX[0] - dotp*rxp[0]/rn;   a2r[1] = dX[1] - dotp*rxp[1]/rn;     a2r[2] = dX[2] - dotp*rxp[2]/rn;                                          
            a2rl = np.sqrt( a2r[0]**2 + a2r[1]**2 + a2r[2]**2 );                                                                         
            r1new[0] = rxp[0] + (1.-cc)*dX[0];   r1new[1] = rxp[1] + (1.-cc)*dX[1];   r1new[2] = rxp[2] + (1.-cc)*dX[2];                                                          
            dotp2  = (r1new[0]-rxp[0])*a2r[0]/a2rl +  (r1new[1]-rxp[1])*a2r[1]/a2rl +  (r1new[2]-rxp[2])*a2r[2]/a2rl;                                                        
            rnew[0] = -r1new[0] + 2*rxp[0] + 2*a2r[0]/a2rl*dotp2;  rnew[1] = -r1new[1] + 2*rxp[1] + 2*a2r[1]/a2rl*dotp2;  rnew[2] = -r1new[2] + 2*rxp[2] + 2*a2r[2]/a2rl*dotp2;
            X[0] = rnew[0];  X[1] = rnew[1];            X[2] = rnew[2];
        else:
            X[0] = dXpX[0];            X[1] = dXpX[1];            X[2] = dXpX[2];

        if nts*click < RT:
            click += 1                 
            SIRfile_tmp = np.zeros(3)
            SIRfile_tmp[0] = X[0]
            SIRfile_tmp[1] = X[1]
            SIRfile_tmp[2] = X[2]
            SIRfile.append(SIRfile_tmp)
    return SIRfile



@njit(parallel = True)
def multiple_loops(N_loops,Tmax,rn,rf,Dmax,Dmin,A1,b):
    SIRfiles = []
    for i in prange(N_loops):
        SIRfile = Single_run_numba(Tmax,rn,rf,Dmax,Dmin,A1,b)
        SIRfiles.append(SIRfile)
    return SIRfiles



N_loops = 8

for testA in range(1):
    for testD in range(1):
        start = time.time()
        
        dt = 0.000005
        Dmax = 1.0
        Dmin = 0.05
        A1 = 1.0
        rn = 1.0
        rf = 0.1 
        Tmax = 100
        b = 1000

        SIRfiles = multiple_loops(N_loops,Tmax,rn,rf,Dmax,Dmin,A1,b)
        end = time.time()
        print(f"\nElapsed (with compilation) = {end - start:.2f}",testD)
        Names = []
        fmt = '%1.4f', '%1.4f', '%1.4f'
        for i0 in range(len(SIRfiles)):
            TT = []
            XN = SIRfiles[i0]
            np.savetxt("TheoreticalData/LPMHeatMapS_Fig3_D%s_A%s_ID%s.txt"%(testD,testA,i0),XN,fmt)

