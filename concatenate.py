
import h5py
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--output','-o',nargs=1,type=str,help='Name of the output file',default=["output.hdf5"])
args = parser.parse_args()




Files    =  os.listdir(os.getcwd())
ndata    = 0
count = 0


for FileName in Files :
    if args.output[0] in FileName : 
        fin       = h5py.File(FileName,"r")
        ndata    += np.array(fin.get("Data"))[:,:].shape[0]
        if count == 0 :
            nfeature  = np.array(fin.get("Data"))[:,:].shape[1]
            neigen    = np.array(fin.get("Eigen_Values"))[:,:].shape[1]
            nocc      = np.array(fin.get("Occupancy"))[:,:].shape[1]
            count += 1
   
        fin.close()

print(ndata)
fout   = h5py.File(args.output[0],"w")
dsetX = fout.create_dataset("Data",(ndata,nfeature),dtype='f')
dsetY = fout.create_dataset("Eigen_Values",(ndata,neigen),dtype='f')
dsetO = fout.create_dataset("Occupancy",(ndata,nocc),'i')
dsetF = fout.create_dataset("Features",(nfeature,),dtype='|S26')


Ntot = 0
for FileName in Files : 
    if args.output[0] in FileName : 
        fin       = h5py.File(FileName,"r")
        data = np.array(fin.get("Data"))[:,:]
        dsetX[Ntot:Ntot+data.shape[0],:] = np.array(fin.get("Data"))[:,:]
        dsetY[Ntot:Ntot+data.shape[0],:] = np.array(fin.get("Eigen_Values"))[:,:]
        dsetO[Ntot:Ntot+data.shape[0],:] = np.array(fin.get("Occupancy"))[:,:]
        features = np.array(fin.get("Features"))
        Ntot += data.shape[0]
        print(data.shape,FileName)
        print(Ntot)
        fin.close()
dsetF[:] = features[:]
print(dsetX)
fout.close()
