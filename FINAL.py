import numpy as np
import matplotlib.pylab as plt
import h5py
import os

f = h5py.File("output.hdf5","r")

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

from sklearn.feature_selection import f_regression, mutual_info_regression

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

for i in range(len(features)):
    print(features[mi_RLX.argsort()][i].decode("utf-8"),mi_RLX[mi_RLX.argsort()][i])

	
# these conditions are according to my system which was azobenzene you should change these conditions
cond1 = np.bitwise_and((mi_RLX>0.097),(mi_RLX<0.104))
cond2 = np.bitwise_and((mi_RLX>0.104),(mi_RLX<0.114))
cond3 = np.bitwise_and((mi_RLX>0.114),(mi_RLX<0.124))
cond4 = np.bitwise_and((mi_RLX>0.124),(mi_RLX<0.243))
cond5 = mi_RLX>0.24
X1 = X_RLX[:,cond1]
f1 = features[cond1]
X2 = X_RLX[:,cond2]
f2 = features[cond2]
X3 = X_RLX[:,cond3]
f3 = features[cond3]
X4 = X_RLX[:,cond4]
f4 = features[cond4]
X5 = X_RLX[:,cond5]
f5 = features[cond5]


from sklearn.decomposition import PCA

# deciding which coordinates are correlated and mixing them with PCA
pca3 = PCA(n_components=1)
pca3.fit(X3)
PCX3 = pca3.transform(X3)

pca4 = PCA(n_components=1)
pca4.fit(X4)
PCX4 = pca4.transform(X4)


X = np.hstack((np.hstack((X1,X2)),np.hstack((PCX3,PCX4)),X5))

from sklearn import neighbors

for i in range(4):
    nn=10
    knn = neighbors.KNeighborsRegressor(nn,weights='uniform')
    knn_fit = knn.fit(X,Y_HOMO_RLX)
    knn2 = neighbors.KNeighborsRegressor(nn,weights='uniform')
    knn2_fit = knn2.fit(X,Y_LUMO_RLX)
    Y_HOMO_RLX = knn_fit.predict(X)
    Y_LUMO_RLX = knn2_fit.predict(X)

from create_init_fin import read_add

INIT_raw = read_add("cis_min.xyz").reshape(-1,1)
FIN_raw  = read_add("trans_min.xyz").reshape(-1,1)

INIT = np.zeros_like(X[0])
FIN =  np.zeros_like(X[0])

INIT[0] = INIT_raw[cond1]
INIT[1] = INIT_raw[cond2]
INIT[2] = pca3.transform(INIT_raw[cond3].reshape(1,-1))
INIT[3] = pca4.transform(INIT_raw[cond4].reshape(1,-1))
INIT[4] = INIT_raw[cond5]

FIN[0] = FIN_raw[cond1]
FIN[1] = FIN_raw[cond2]
FIN[2] = pca4.transform(FIN_raw[cond3].reshape(1,-1))
FIN[3] = pca4.transform(FIN_raw[cond4].reshape(1,-1))
FIN[4] = FIN_raw[cond5]


INIT = INIT.reshape(1,-1)
FIN = FIN.reshape(1,-1)

X = np.vstack((X,INIT))
Y_HOMO_RLX = np.append(Y_HOMO_RLX,knn_fit.predict(INIT))
Y_LUMO_RLX = np.append(Y_LUMO_RLX,knn2_fit.predict(INIT))
X = np.vstack((X,FIN))
Y_HOMO_RLX = np.append(Y_HOMO_RLX,knn_fit.predict(FIN))
Y_LUMO_RLX = np.append(Y_LUMO_RLX,knn2_fit.predict(FIN))


colors = ['#00FFFF','#7FFFD4','#F5F5DC', '#FFE4C4','#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00','#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B','#2F4F4F','#00CED1','#9400D3','#FF1493','#00BFFF','#696969','#1E90FF','#B22222','#228B22','#FF00FF','#DCDCDC','#FFD700','#DAA520','#808080','#008000','#ADFF2F','#FF69B4','#CD5C5C','#F0E68C','#7CFC00','#FFFACD','#ADD8E6','#F08080','#FAFAD2','#90EE90','#D3D3D3','#FFB6C1','#FFA07A','#20B2AA','#87CEFA','#778899','#B0C4DE','#FFFFE0','#00FF00','#32CD32','#FF00FF','#800000','#66CDAA','#0000CD','#BA55D3','#9370DB','#3CB371','#7B68EE','#00FA9A','#48D1CC','#C71585','#191970','#FFE4B5','#FFDEAD','#000080','#808000','#6B8E23','#FFA500','#FF4500','#DA70D6','#EEE8AA','#98FB98','#AFEEEE','#DB7093','#FFEFD5','#FFDAB9','#CD853F','#FFC0CB','#DDA0DD','#B0E0E6','#800080','#FF0000','#BC8F8F','#4169E1','#8B4513','#FA8072','#FAA460','#2E8B57','#A0522D','#C0C0C0','#87CEEB','#6A5ACD','#708090','#00FF7F','#4682B4','#D2B48C','#008080','#D8BFD8','#FF6347','#40E0D0','#EE82EE','#F5DEB3','#FFFF00','#9ACD32']
plt.figure(1,figsize=(20,20))
plt.scatter(X[:,4],Y_HOMO_RLX,c=colors[11])
plt.xlim(0,180)
plt.savefig("datapoints.png")

import networkx as nx
import scipy.sparse
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, 50, mode='connectivity', include_self=False)
G = nx.from_scipy_sparse_matrix(connectivity,create_using=nx.DiGraph())
#new_MI = np.array([ 0.09719257,  0.10492396,  0.13380462,  0.13790951,  0.24386509])
new_MI = mutual_info_regression(X, Y_HOMO_RLX)

pos = {}
for inode in range(len(X)):
    pos[inode] = (X[inode,4],Y_HOMO_RLX[inode])

for n,p in pos.items():
    G.node[n]['pos']=p

edges = G.edges()
faulty = []
threshold = np.sqrt(np.dot(new_MI,np.square(np.array([4,4,4,4,4]))))
for iedge in range(len(edges)):
    nn = edges[iedge][1]
    n = edges[iedge][0]
    deltaE = (Y_HOMO_RLX[nn]-Y_HOMO_RLX[n])
    deltaX = np.sqrt(np.dot(new_MI,np.square(X[nn]-X[n])))*np.sign(np.dot(new_MI,X[nn]-X[n]))
    w = np.exp(deltaE/deltaX)
    #w = np.exp(deltaE)
    if (abs(deltaE) > 0.075) or (deltaX>threshold):
        faulty.append((n,nn))
    G.edge[n][nn]['weight'] = w

G.remove_edges_from(faulty)

ndata = len(X)
path = nx.dijkstra_path(G,source=ndata-2,target=ndata-1)
print(path)
np.savetxt("path",path,fmt='%i')
EN_MAX = np.argmin(-1*Y_HOMO_RLX[path])

#X[path,0].reshape(-1,1).shape

#plt.figure(2,figsize=(10,10))
f,ax = plt.subplots(6,sharex=True,figsize=(20,20))
ax[1].plot(X[path,4],label=f5[0].decode("utf-8"),linewidth=4)
#ax[1].set_title("MI1-"+f5[0].decode("utf-8"),fontsize=20)
ax[1].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[1].set_yticklabels([' ','20',' ','60',' ','100',' ','140',' ','180'], fontsize = 20)
ax[1].axhline(y=X[path,4][EN_MAX], color='k', linestyle='--',linewidth=3)
ax[1].legend(fontsize=20,shadow=True, fancybox=True,loc=7)
ax[2].plot(pca4.inverse_transform(X[path,3].reshape(-1,1))[:,1],label=f4[0].decode("utf-8"),linewidth=4)
#ax[2].set_title("MI2-"+f4[0].decode("utf-8"),fontsize=20)
ax[2].set_yticklabels([' ','5',' ','15',' ','25',' ','35',' '], fontsize = 20)
ax[2].legend(fontsize=20,shadow=True, fancybox=True)
ax[3].plot(pca3.inverse_transform(X[path,2].reshape(-1,1))[:,1],label=f3[0].decode("utf-8"),linewidth=4)
#ax[3].set_title("MI3-"+f3[0].decode("utf-8"),fontsize=20)
ax[3].set_yticklabels([' ','145',' ','155',' ','165',' ','175',' ','185'], fontsize = 20)
ax[3].legend(fontsize=20,shadow=True, fancybox=True,loc=7)
ax[4].plot(X[path,1],label="MI4-"+f2[0].decode("utf-8"),linewidth=4)
#ax[4].set_title("MI4-"+f2[0].decode("utf-8"),fontsize=20)
ax[4].set_yticklabels([' ','105',' ','115',' ','125',''], fontsize = 20)
ax[4].legend(fontsize=20,shadow=True, fancybox=True)
ax[5].plot(X[path,0],label=f1[0].decode("utf-8"),linewidth=4)
#ax[5].set_title()
ax[5].legend(fontsize=20,shadow=True, fancybox=True)
ax[5].set_yticklabels([' ','115',' ','125',' ','135'], fontsize = 20)
ax[0].plot(Y_HOMO_RLX[path],label="HOMO",linewidth=4)
#ax[0].set_title(,fontsize=15)
ax[0].legend(fontsize=20,shadow=True, fancybox=True)
ax[0].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[0].axhline(y=Y_HOMO_RLX[path][EN_MAX], color='k', linestyle='--',linewidth=3)
ax[0].set_yticklabels([-7.0,' ',-6.6,' ',-6.2,' ',-5.8,' '], fontsize = 20)
#ax[6].plot(w)
#ax[6].set_title("weights")
#plt.plot(pca1.inverse_transform(X[path,0])[0],Y_HOMO_RLX[path])
plt.savefig("plot0.png")

f,ax = plt.subplots(2,sharex=True,figsize=(20,20))
ax[1].plot(X[path,4],label=f5[0].decode("utf-8"),linewidth=4)
ax[1].plot(pca4.inverse_transform(X[path,3].reshape(-1,1))[:,1],label=f4[0].decode("utf-8"),linewidth=4)
ax[1].plot(pca3.inverse_transform(X[path,2].reshape(-1,1))[:,1],label=f3[0].decode("utf-8"),linewidth=4)
ax[1].plot(X[path,1],label=f2[0].decode("utf-8"),linewidth=4)
ax[1].plot(X[path,0],label=f1[0].decode("utf-8"),linewidth=4)
ax[1].legend(fontsize=20,loc=7)
ax[1].yaxis.set_ticks(np.arange(0,190,20))
ax[1].set_ylim(0,180)
ax[1].tick_params(labelsize=20)
ax[1].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[1].axhline(y=X[path,4][EN_MAX], color='k', linestyle='--',linewidth=3)

ax[0].plot(Y_HOMO_RLX[path],label="HOMO",linewidth=4)
ax[0].legend(fontsize=20)
ax[0].tick_params(labelsize=20)
ax[0].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[0].axhline(y=Y_HOMO_RLX[path][EN_MAX], color='k', linestyle='--',linewidth=3)
plt.savefig("plot1.png")


f,ax = plt.subplots(3,sharex=True,figsize=(20,20))
ax[1].plot(X[path,4],label=f5[0].decode("utf-8"),linewidth=4)
ax[1].legend(fontsize=20,loc=7)
ax[1].yaxis.set_ticks(np.arange(0,190,20))
ax[1].set_ylim(0,180)
ax[1].tick_params(labelsize=20)
ax[1].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[1].axhline(y=X[path,4][EN_MAX], color='k', linestyle='--',linewidth=3)

ax[2].plot(pca4.inverse_transform(X[path,3].reshape(-1,1))[:,1],label=f4[0].decode("utf-8"),linewidth=4)
ax[2].plot(pca3.inverse_transform(X[path,2].reshape(-1,1))[:,1],label=f3[0].decode("utf-8"),linewidth=4)
ax[2].plot(X[path,1],label=f2[0].decode("utf-8"),linewidth=4)
ax[2].plot(X[path,0],label=f1[0].decode("utf-8"),linewidth=4)
ax[2].legend(fontsize=20,loc=7)
ax[2].yaxis.set_ticks(np.arange(0,190,20))
ax[2].set_ylim(0,180)
ax[2].tick_params(labelsize=20)

ax[0].plot(Y_HOMO_RLX[path],label="HOMO",linewidth=4)
ax[0].legend(fontsize=20)
ax[0].tick_params(labelsize=20)
ax[0].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[0].axhline(y=Y_HOMO_RLX[path][EN_MAX], color='k', linestyle='--',linewidth=3)
plt.savefig("plot2.png")


f,ax = plt.subplots(2,sharex=True,figsize=(20,20))
ax[1].plot(X[path,4],label=f5[0].decode("utf-8"),linewidth=4)
ax[1].legend(fontsize=20,loc=7)
ax[1].yaxis.set_ticks(np.arange(0,190,20))
ax[1].set_ylim(0,180)
ax[1].tick_params(labelsize=20)
ax[1].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[1].axhline(y=X[path,4][EN_MAX], color='k', linestyle='--',linewidth=3)

ax[0].plot(Y_HOMO_RLX[path],label="HOMO",linewidth=4)
ax[0].legend(fontsize=20)
ax[0].tick_params(labelsize=20)
ax[0].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[0].axhline(y=Y_HOMO_RLX[path][EN_MAX], color='k', linestyle='--',linewidth=3)
plt.savefig("plot4.png")


f,ax = plt.subplots(2,sharex=True,figsize=(20,20))

ax2 = ax[0].twinx()

ax[0].plot(Y_HOMO_RLX[path],label="HOMO",linewidth=4,color='g')
#ax[0].legend(fontsize=20)
ax[0].tick_params(labelsize=20)
ax[0].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[0].axhline(y=Y_HOMO_RLX[path][EN_MAX], color='k', linestyle='--',linewidth=3)
ax2.plot(X[path,4],linewidth=4,color='b')

ax[0].set_xlabel('Path Parameter')
ax[0].set_ylabel('Energy (eV)', fontsize=24,color='b')
ax2.set_ylabel(f5[0].decode("utf-8"), fontsize=24,color='g')
#ax2.legend(fontsize=20)

ax[1].plot(pca4.inverse_transform(X[path,3].reshape(-1,1))[:,1],label=f4[0].decode("utf-8"),linewidth=4)
ax[1].plot(pca3.inverse_transform(X[path,2].reshape(-1,1))[:,1],label=f3[0].decode("utf-8"),linewidth=4)
ax[1].plot(X[path,1],label=f2[0].decode("utf-8"),linewidth=4)
ax[1].plot(X[path,0],label=f1[0].decode("utf-8"),linewidth=4)
ax[1].legend(fontsize=20,loc=7)
ax[1].yaxis.set_ticks(np.arange(0,190,20))
ax[1].set_ylim(0,180)
ax[1].tick_params(labelsize=20)
ax[1].axvline(x=EN_MAX, color='k', linestyle='--',linewidth=3)
ax[1].axhline(y=X[path,4][EN_MAX], color='k', linestyle='--',linewidth=3)

plt.savefig("plot4.png")
