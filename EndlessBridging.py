from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
import time


@njit
def Single_run_numba(Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn,L):
############################## INITIALIZATION ############################################

    SIRfile = []
    x = 0; y = 0; z = 0;
    RT = 0.0;
    rv = 0.035;

    rho = 200/(4/3.0*3.141592*0.1**3);
    click = 0; 
    clickLive = 0
    dt = 0.000001; DB = DBF
    nts = dt*1000; Ntot = round(100.0/dt); 
    Bin = np.zeros((100,3)); Nbin = 0; 
    ci = 0;
    PIB = 0; checkLive = 0; BoundID = 0;BoundVec = np.zeros(3);
    binvec = np.zeros(3); suggvec = np.zeros(3);NewX = np.zeros(3); drvec = np.zeros(3);
    while RT < Tmax:
        ci = ci+1;
        RT += dt
        cll = 0;
        xm1 = x; ym1 = y; zm1 = z;

############# Move Binding sites
        for i2 in range(Nbin):            
            dBx = np.sqrt(2*DB*dt)*np.random.normal();
            dBy = np.sqrt(2*DB*dt)*np.random.normal()
            dBz = np.sqrt(2*DB*dt)*np.random.normal()
            Bin[i2,0] = Bin[i2,0] + dBx;
            Bin[i2,1] = Bin[i2,1] + dBy;
            Bin[i2,2] = Bin[i2,2] + dBz;

            for i3 in range(Nbin):
                rbin = np.sqrt( (Bin[i2,0]-Bin[i3,0])**2 + (Bin[i2,1]-Bin[i3,1])**2 + (Bin[i2,2]-Bin[i3,2])**2)
                if (rbin < 2*rb and i2 != i3):
                    binvec[0] = (Bin[i2,0]-Bin[i3,0])/rbin;
                    binvec[1] = (Bin[i2,1]-Bin[i3,1])/rbin;
                    binvec[2] = (Bin[i2,2]-Bin[i3,2])/rbin;
                    binr = 2*rb - rbin;
                    Bin[i2,0] = Bin[i3,0] + binvec[0]*(2*rb+binr);
                    Bin[i2,1] = Bin[i3,1] + binvec[1]*(2*rb+binr);
                    Bin[i2,2] = Bin[i3,2] + binvec[2]*(2*rb+binr);
            
############# Particle Binding sites
        if (PIB == 0):
            dx = np.sqrt(2*D*dt)*np.random.normal(); 
            dy = np.sqrt(2*D*dt)*np.random.normal(); 
            dz = np.sqrt(2*D*dt)*np.random.normal();
            x1 = x + dx;
            y1 = y + dy;
            z1 = z + dz;

        else:
            x1 = Bin[BoundID,0] + BoundVec[0];
            y1 = Bin[BoundID,1] + BoundVec[1];
            z1 = Bin[BoundID,2] + BoundVec[2];
            dx = x1-x; dy = y1-y; dz = z1-z;
            if (np.random.random() < Koff*dt):
                PIB = 0;

        i2 = 0;
        while i2 < Nbin:
            rsugg = np.sqrt( (Bin[i2,0]-x1)**2 + (Bin[i2,1]-y1)**2 + (Bin[i2,2]-z1)**2)
            if (rsugg < rb and PIB == 0):
                suggvec[0] = (x1-Bin[i2,0])/rsugg
                suggvec[1] = (y1-Bin[i2,1])/rsugg
                suggvec[2] = (z1-Bin[i2,2])/rsugg
                delr = rb-rsugg;
                NewX[0] = Bin[i2,0] + suggvec[0]*(rb+delr);
                NewX[1] = Bin[i2,1] + suggvec[1]*(rb+delr);
                NewX[2] = Bin[i2,2] + suggvec[2]*(rb+delr);
                x1 = NewX[0]; y1 = NewX[1]; z1 = NewX[2];
                pacc = k0*np.sqrt(3.141592*dt/D);
                if np.random.random() < pacc:
                    PIB = 1;
                    BoundID = i2;
                    BoundVec[0] = suggvec[0]*(rb+delr);
                    BoundVec[1] = suggvec[1]*(rb+delr);
                    BoundVec[2] = suggvec[2]*(rb+delr);

            if (rsugg > rv):
                for j2 in range(i2,Nbin):
                    Bin[j2,0] = Bin[j2+1,0];
                    Bin[j2,1] = Bin[j2+1,1];
                    Bin[j2,2] = Bin[j2+1,2];
                Nbin = Nbin-1;
                if (PIB==1 and BoundID > i2):
                    BoundID = BoundID-1;                
            else:
                i2+=1;
        
        d = np.sqrt(dx**2 + dy**2 + dz**2);
        V = 4*3.141592/3*rv**3 - 1.0/12*3.141592*(4*rv+d)*(2*rv-d)**2;
        p = V*rho;

        if (DB > 0.00001):
            drvec[0] = x1-x
            drvec[1] = y1-y
            drvec[2] = z1-z
            drdr = np.sqrt( (x1-x)**2 + (y1-y)**2 + (z1-z)**2)
            xri = (x+x1)/2.0 + drvec[0]/drdr*rv
            yri = (y+y1)/2.0 + drvec[1]/drdr*rv
            zri = (z+z1)/2.0 + drvec[2]/drdr*rv
        
        if (np.random.random() < p):
            Nbin = Nbin+1;
            acc = 0;
            cacc = 0
            while acc == 0:
                cacc += 1
                rsugg = (rv**3 + (d**3)*(np.random.random()-1.0))**(1.0/3.0);
                theta = 2*3.141592*np.random.random();
                phi = np.arccos(1 - 2*np.random.random());
                xnew = x1 + rsugg*np.sin(phi)*np.cos(theta);
                ynew = y1 + rsugg*np.sin(phi)*np.sin(theta);
                znew = z1 + rsugg*np.cos(phi);
                rold = np.sqrt( (x-xnew)**2+(y-ynew)**2+(z-znew)**2);
                rnew = np.sqrt( (x1-xnew)**2+(y1-ynew)**2+(z1-znew)**2);
                if (rold > rv and rnew <= rv):
                    acc = 1;
                    for j in range(Nbin-1):
                        if np.sqrt( (xnew-Bin[j,0])**2 + (ynew-Bin[j,1])**2 + (znew-Bin[j,2])**2) < 2*rb:                            
                            acc = 0;
                    if (acc == 1):
                        Bin[Nbin,0] = xnew;
                        Bin[Nbin,1] = ynew;
                        Bin[Nbin,2] = znew;
        x = x1; y = y1; z = z1;
        
    
        while nts*click < RT:
            chit = 0
            cin = 0
            SIRfile_tmp = np.zeros(3)
            SIRfile_tmp[0] = x
            SIRfile_tmp[1] = y
            SIRfile_tmp[2] = z
            SIRfile.append(SIRfile_tmp)
            click += 1

        if RT > clickLive:
            clickLive+=1
            print(clickLive)
    return SIRfile



#header = ['Time','S1', 'S2', 'S3']
@njit(parallel = True)
def multiple_loops(N_loops,Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn,L):
    SIRfiles = []
    for i in prange(N_loops):
        SIRfile = Single_run_numba(Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn,L)
        SIRfiles.append(SIRfile)

    return SIRfiles

N_loops = 2
TN = 0
for in1 in range(4):
    for in2 in range(1):
        for in3 in range(1):
            start = time.time()
            DBF = 0.005
            k0 = 10.0**(in1-1)
            dt = 0.000001
            pest = 0.1429
            D = 1.0
            L = 1;
            rn = 0.9
            rf = 0.1
            rb = 0.01
            rho = 0.2
            Koff = pest/(1.0-pest)*4*3.141592*D*rb/(1+D/(k0*rb))*200.0/(4*3.141592/3.0*0.1**3)
            NB = int(rho*rf**3.0/rb**3.0)
            Tmax = 100
            TN += 1
            rtnn = TN

            SIRfiles = multiple_loops(N_loops,Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn,L)
            end = time.time()
            print(f"\nElapsed (with compilation) = {end - start:.2f}",TN)

            for i0 in range(len(SIRfiles)):
                DataTmp = []
                XN = SIRfiles[i0]
                for j in range(len(XN)):
                    x1 = XN[j][0];
                    y1 = XN[j][1];
                    z1 = XN[j][2];
                    DX = [x1,y1,z1]
                    DataTmp.append(DX)
                    
                fmt = '%1.6f', '%1.6f', '%1.6f'
                np.savetxt("TheoreticalData/ForeverBridge_NewAttemptV2_test%s_ID%s.txt"%(in1+1,2+i0),DataTmp,fmt)

