import numpy as np
import matplotlib.pylab as plt
import h5py
import os
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn import neighbors

f = h5py.File("MY_DATA.hdf5","r")

X = np.array(f.get("Data"))
Y = np.array(f.get("Eigen_Values"))
features = np.array(f.get("Features"))
occupancy = np.array(f.get("Occupancy"))

ndata = len(X)
idx_exc = occupancy[:,1] == 1
idx_rlx = np.bitwise_not(idx_exc)
Y_HOMO_EXC = Y[idx_exc,0]
Y_LUMO_EXC = Y[idx_exc,1]
X_EXC = X[idx_exc]
Y_HOMO_RLX = Y[idx_rlx,0]
Y_LUMO_RLX = Y[idx_rlx,1]
X_RLX = X[idx_rlx]

print("LOADING OR CALCULATING MI")
if os.path.exists("MI.hdf5"):
    f2 = h5py.File("MI.hdf5","r")
    mi_RLX = np.array(f2.get("MI"))
    features = np.array(f2.get("features"))
else : 
    mi_RLX = mutual_info_regression(X_RLX, Y_HOMO_RLX)
    f2 = h5py.File("MI.hdf5")
    dset1 = f2.create_dataset("MI",mi_RLX.shape,dtype = "f")
    dset2 = f2.create_dataset("features",features.shape,dtype = "|S26")
    dset1[:] = mi_RLX[:]
    dset2[:] = features[:]
    
X_reduced = X_RLX[:,mi_RLX > 0.179]
features_reduced = features[mi_RLX > 0.179]

print("reduced data from"+str(X_RLX.shape)+" to "+str(X_reduced.shape))
nn=4
knn = neighbors.KNeighborsRegressor(nn,weights='uniform')
knn_fit = knn.fit(X_reduced,Y_HOMO_RLX)
nfeature = len(features_reduced)
mins = np.zeros((nfeature,))
maxs = np.zeros((nfeature,))
for ifeat in range(nfeature):
    mins[ifeat] = np.min(X_reduced[:,ifeat])
    maxs[ifeat] = np.max(X_reduced[:,ifeat])
T = X_reduced
y_predict = knn_fit.predict(T)
ndata = len(T)
threshold = np.ones_like(T[0])*2
nloop = 0
while (ndata > 20000 and nloop < 10) :
    pointer = np.zeros((len(T[:,0]),))
    ndata = len(T)
    idata = np.random.randint(ndata,size=1)
    for jdata in range(ndata):
        if np.all(abs(T[idata]-T[jdata]) <= threshold):
            pointer[jdata] = 1
        else :
            pointer[jdata] = 0
    if sum(pointer) !=1 :
        nloop = 0
        x_datum = np.average(T[pointer == 1],axis=0)
        y_datum  = np.average(y_predict[pointer == 1])
        T = T[pointer == 0]
        y_predict = y_predict[pointer == 0]
        T = np.vstack((T,x_datum))
        y_predict = np.append(y_predict,y_datum)
    else :
        nloop += 1
    print(ndata,sum(pointer))
    


f3 = h5py.File("REDUCED"+str(len(T))+".hdf5","w")
dset1 = f3.create_dataset("X",T.shape,dtype='f')
dset2 = f3.create_dataset("Y",y_predict.shape,dtype='f')
dset3 = f3.create_dataset("features",features_reduced.shape,dtype = '|S26')
dset1[:,:] = T[:,:]
dset2[:] = y_predict[:]
dset3[:] = features_reduced[:]
