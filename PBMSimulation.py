from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
import time


@njit
def Single_run_numba(Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn):
############################## INITIALIZATION ############################################
    nts = 0.02
    rin = rb;
    npi = 3.141592
    p = k0*np.sqrt(3.141592/D*dt); 
    PP = np.zeros((NB, 3))
    Boxes = np.zeros((21,21,21,10), dtype=np.uint32)
    nBoxes = np.zeros((21,21,21), dtype=np.uint32)
    hh = (rf+rb)/10.0
    PosBox = np.zeros((NB,3))
    SIRfile = []
    for in1 in range(NB):
        acc = 0
        while (acc == 0):
            ra0 = rf - 2*rf*np.random.rand()
            ra1 = rf - 2*rf*np.random.rand()
            ra2 = rf - 2*rf*np.random.rand()
            if (np.sqrt(ra0**2 + ra1**2 +ra2**2) < rf):
                act = 0
                for i2 in range(in1):
                    if (np.sqrt((ra0-PP[i2,0])**2 + (ra1-PP[i2,1])**2 +(ra2-PP[i2,2])**2) < 2.01*rb):
                        act = 1
                if(act == 0):
                    acc = 1
                    PP[in1,0] = ra0
                    PP[in1,1] = ra1
                    PP[in1,2] = ra2
                    nx0 = ra0/hh;
                    if (nx0 < 0):
                        nx = 10+np.ceil(nx0)
                    else:
                        nx = 10+np.floor(nx0)
                    ny0 = ra1/hh;
                    if (ny0 < 0):
                        ny = 10+np.ceil(ny0)
                    else:
                        ny = 10+np.floor(ny0)
                    nz0 = ra2/hh;
                    if (nz0 < 0):
                        nz = 10+np.ceil(nz0)
                    else:
                        nz = np.int(10+np.floor(nz0))
                        
                    nx = np.int(nx)
                    ny = np.int(ny)
                    nz = np.int(nz)
                    Boxes[nx,ny,nz,nBoxes[nx,ny,nz]]
                    nBoxes[nx,ny,nz] += 1
                    PosBox[in1,0] = nx
                    PosBox[in1,1] = ny
                    PosBox[in1,2] = nz



    InfPoint = np.zeros(3)
    InfPointN = np.zeros(3)
    NormV = np.zeros(3)
    dr = np.zeros(3)
    X0 = np.zeros(3)
    X1 = np.zeros(3)
    cx = np.zeros(3)
    cx0 = np.zeros(3)
    cc = 0.0
    X = np.ones(3)*0.001 
    rtn1 = rn*np.random.rand()


    theta = 2 * npi *np.random.rand();
    phi = np.arccos(1 - 2 *np.random.rand());
    x = rtn1*np.sin(phi)*np.cos(theta);
    y = rtn1*np.sin(phi)*np.sin(theta);
    z = rtn1*np.cos(phi);
    PosX = np.zeros(3, dtype=np.uint32)
    for i2 in range(NB):
        if ( np.sqrt((PP[i2,0]-x)**2 + (PP[i2,1]-y)**2 + (PP[i2,2]-z)**2) < rb):
            ParticleIsBound = 1
            BoundID = i2
            theta = 2 * npi *np.random.rand();
            phi = np.arccos(1 - 2 *np.random.rand());
            XBO1 = rtn1*np.sin(phi)*np.cos(theta);
            XBO2 = rtn1*np.sin(phi)*np.sin(theta);
            XBO3 = rtn1*np.cos(phi);
    click = 0
    RT = 0

    nrep = 0.001; clickrep = 0;
    CheckLive = 0; bt = 0;
############################## RUN SCRIPT    ############################################
    while(RT < Tmax):
        if (ParticleIsBound == 1):
            for in1 in range(NB):
                dbx = np.sqrt(2*DBF*dt)*np.random.normal()
                dby = np.sqrt(2*DBF*dt)*np.random.normal()
                dbz = np.sqrt(2*DBF*dt)*np.random.normal()                                                                                                       
                btx = PP[in1,0] + dbx                                                                                                                    
                bty = PP[in1,1] + dby
                btz = PP[in1,2] + dbz
                if (np.sqrt(btx**2 + bty**2 + btz**2) < rf):
                    PP[in1,0] = btx
                    PP[in1,1] = bty
                    PP[in1,2] = btz
                X[0] = PP[BoundID,0]+XBO1
                X[1] = PP[BoundID,1]+XBO2
                X[2] = PP[BoundID,2]+XBO3
                
            if (Koff*dt > np.random.rand()):                    
                ParticleIsBound = 0;


        else:
            x0 = X[0]
            y0 = X[1]
            z0 = X[2]            
            r0 = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)

            dt = 0.000001
            dx = np.sqrt(2*D*dt)*np.random.normal();
            dy = np.sqrt(2*D*dt)*np.random.normal();
            dz = np.sqrt(2*D*dt)*np.random.normal();

            X[0] += dx;
            X[1] += dy;
            X[2] += dz;

            xn = X[0]; yn = X[1]; zn = X[2];
            rw = np.sqrt(xn*xn + yn*yn + zn*zn)
        
            if rw > rn: ### Reflect at nucleus boundary
                cc = (rn-r0)/(rw-r0);
                dr[0] = dx;            dr[1] = dy;            dr[2] = dz
                X1[0] = xn;            X1[1] = yn;            X1[2] = zn;
                InfPoint[0] = x0 + dr[0]*cc;            InfPoint[1] = y0 + dr[1]*cc;            InfPoint[2] = z0 + dr[2]*cc
                NormV[0] = (InfPoint[0]-cx0[0]);
                NormV[1] = (InfPoint[1]-cx0[1]);
                NormV[2] = (InfPoint[2]-cx0[2]);
                NormV = NormV/np.sqrt(NormV[0]*NormV[0] + NormV[1]*NormV[1] + NormV[2]*NormV[2]);
                DotIV = (X1[0]-InfPoint[0])*NormV[0] + (X1[1]-InfPoint[1])*NormV[1] + (X1[2]-InfPoint[2])*NormV[2]
                X[0] = X1[0] - 2*NormV[0]*(DotIV);
                X[1] = X1[1] - 2*NormV[1]*(DotIV);
                X[2] = X1[2] - 2*NormV[2]*(DotIV);
            elif (rw < (rf+rb)): ### Search for spheres to interact with
                for in1 in range(NB):
                    dbx = np.sqrt(2*DBF*dt)*np.random.normal()                                                                                  
                    dby = np.sqrt(2*DBF*dt)*np.random.normal()
                    dbz = np.sqrt(2*DBF*dt)*np.random.normal()                                                                                                       
                    btx = PP[in1,0] + dbx   
                    bty = PP[in1,1] + dby                                
                    btz = PP[in1,2] + dbz
                    if (np.sqrt(btx**2 + bty**2 + btz**2) < rf):
                        PP[in1,0] = btx
                        PP[in1,1] = bty
                        PP[in1,2] = btz

                    cx[0] = PP[in1,0]
                    cx[1] = PP[in1,1]
                    cx[2] = PP[in1,2]
                    rw = np.sqrt( (xn-cx[0])**2 + (yn-cx[1])**2 +(zn-cx[2])**2);
                    r0 = np.sqrt( (x0-cx[0])**2 + (y0-cx[1])**2 +(z0-cx[2])**2);                
                    if rw < rb: ### If the particle is whithin a sphere
                        rk = np.sqrt( (X[0]-cx[0])**2 + (X[1]-cx[1])**2 + (X[2]-cx[2])**2)
                        XBO1 = (X[0]-cx[0])/rk*(2*rb-rw)
                        XBO2 = (X[1]-cx[1])/rk*(2*rb-rw)
                        XBO3 = (X[2]-cx[2])/rk*(2*rb-rw)                    
                        if (np.random.rand() < p): ### here we bind                        
                            BoundID = in1
                            ParticleIsBound = 1;
                            bt = RT;
                        X[0] = cx[0]+XBO1
                        X[1] = cx[1]+XBO2
                        X[2] = cx[2]+XBO3
        RT += dt;


        if (RT > CheckLive):
            CheckLive+=1 
            RR = np.sqrt(PP[:,0]**2 + PP[:,1]**2 + PP[:,2]**2)
            print(RT,rtnn)


        while nts*click < RT:
            chit = 0
            cin = 0
            SIRfile_tmp = np.zeros(3)
            SIRfile_tmp[0] = X[0]
            SIRfile_tmp[1] = X[1]
            SIRfile_tmp[2] = X[2]
            SIRfile.append(SIRfile_tmp)
            click += 1

    return SIRfile



#header = ['Time','S1', 'S2', 'S3']
@njit(parallel = True)
def multiple_loops(N_loops,Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn):
    SIRfiles = []
    for i in prange(N_loops):
        SIRfile = Single_run_numba(Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn)
        SIRfiles.append(SIRfile)

    return SIRfiles

N_loops = 8
TN = 0
for in1 in range(1):
    for in2 in range(1):
        for in3 in range(1):
            start = time.time()
            DBF = 0.005
            k0 = 100
            Koff = 500
            rf = 0.1
            rn = 1.0
            dt = 0.000001
            pest = 0.1429
            D = 1.0
            L = 1;
            rb = 0.01
            rho = 0.2
            NB = int(rho*rf**3.0/rb**3.0)            
            Tmax = 300
            TN += 1
            rtnn = TN

            SIRfiles = multiple_loops(N_loops,Tmax,DBF,k0,dt,Koff,NB,D,rn,rf,rb,rtnn)
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
                np.savetxt("TheoreticalData/FitPBM_Final2DB05_test%s_ID%s.txt"%(TN,i0),DataTmp,fmt)

