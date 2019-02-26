import numpy as np
#import matplotlib.pylab as plt
import h5py
import os
import argparse
from sklearn.feature_selection import  mutual_info_regression
#import gc
#import resource

#def mem():
#    print('Memory usage         : % 2.2f MB' % round(
#        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0,1)
#    )

#mem()

parser = argparse.ArgumentParser()
parser.add_argument('--nstate',nargs=1,type=int,help='KS energy state index',default=[0])
parser.add_argument('--output','-o',nargs=1,type=str,help='Name of the output file',default=["MI.hdf5"])
parser.add_argument('--input','-i',nargs=1,type=str,help='Name of input file')
parser.add_argument('--nsample','-ns',nargs=1,type=int,help='number of data points to be sampled from the data')
parser.add_argument('--nrun','-nr',nargs=1,type=int,help='number of times to sample data',default=[10])
parser.add_argument('--nprocess','-np',nargs=1,type=int,help='number processors',default=[1])

args = parser.parse_args()
    
fin = h5py.File(args.input[0],"r")
q = np.array(fin.get("Data"))
E = np.array(fin.get("Eigen_Values"))[:,args.nstate[0]-1]
features = np.array(fin.get("Features"))
nfeature = len(features)
occupancy = np.array(fin.get("Occupancy"))
fin.close()
    
ndata = len(q)
if args.nsample == None :
    args.nsample = [ndata/10]
        
fout   = h5py.File(args.output[0],"w")
dsetMI = fout.create_dataset("MI",(args.nrun[0],nfeature),dtype='f')
dsetF  = fout.create_dataset("Features",(nfeature,),dtype='|S26')
dsetMean = fout.create_dataset("MI.means",(nfeature,),dtype='f')
dsetStd = fout.create_dataset("MI.std",(nfeature,),dtype='f')
    
dsetF[:] = features[:]
    
IndexRelax = (occupancy[:,1] == 0)
E = E[IndexRelax]
q = q[IndexRelax]
    


IndexNaN = np.sum(np.isnan(q),axis=1) == 0
E = E[IndexNaN]
q = q[IndexNaN]

NAN = np.sum(np.isnan(q),axis=1)
temp_MI = np.zeros((args.nrun[0],nfeature))


for irun in range(args.nrun[0]):
    sample = np.random.randint(0,len(E),args.nsample[0])
    print('irun=',irun)
    print('q[sample].shape=',q[sample].shape)
    MI = mutual_info_regression(q[sample],E[sample])
    temp_MI[irun,:] = MI[:]
#    mem()
dsetMI[:,:] = temp_MI[:,:]
dsetMean[:] = np.mean(temp_MI,axis=0)
dsetStd[:] = np.std(temp_MI,axis=0)
fout.close()


#    p = mp.Pool(args.nprocess[0])
#    results = p.map(worker,arg)
#    for irun in range(len(results)):
#        temp_MI[irun,:] = results[irun][:]
#        print(results[irun][:])

