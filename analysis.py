import sys
import numpy as np
import re
import itertools
import multiprocessing as mp
import h5py
import os

def angle(n1,n2) : 						# calculates the angles between two vectors
    dot = np.dot(n1,n2)				    	# dot product of the vectors
    mag1 = np.sqrt(np.dot(n1 ,n1))  			# Magnitude of n1 
    mag2 = np.sqrt(np.dot(n2 ,n2))		  		# Magnitude of n2
    ang_r = np.arccos(dot/(mag1*mag2))  			# Angle between normals using Arccos in radians
    ang_d = np.rad2deg(ang_r) 				# changing the angle to Degrees)
    return ang_d

#############################################################################
# This fuction finds the normal vector to the plane using 3 points in plane #
#############################################################################

def norm2plane(p1,p2,p3) : 					# finding the normal vector to a plane having 3 points in that plane
    v1 = p3 - p1 						# finding the 1st vector in the plane
    v2 = p2 - p1 						# finding the second vector in the plane
    cp = np.cross(v1,v2) 					# finding the cross product of v1 and v2 to get a vector orthogonal
    cp = cp/np.sqrt(np.dot(cp,cp)) 	                        # normalizing the voctor
    return cp


#######################################################################################################
# This function calculates the dihedral angle between two planes, plane1 includes atom1, atom2, atom3 # 
# and plane2 includes atom2, atom3, atom 4 							      #
#######################################################################################################

def dihedral(a1,a2,a3,a4) :					# finds the dihedral angle for 4 positions in space
    n1=norm2plane(a1,a2,a3)					# finding the normal vector for the 1st 3 positions
    n2=norm2plane(a4,a2,a3)					# finging the normal vector for the 2nd 3 positions
    return angle(n1,n2)					# returning the angle between the two normal vectors

def Tangle(a1,a2,a3) :
    v1 = a1 - a2
    v2 = a3 - a2
    return angle(v1,v2)

def dist(a,b) : 
    return np.linalg.norm(a-b)

def find(vector,number):
    match = []
    for i in range(len(vector)):
        if vector[i] == number :
            match.append(i)
    return match

def initializer(atom_names ,structure):
    # the following loop calculates all the distances in the structure
    distances = np.zeros((natom,natom))
    for iatom in range(natom) : 
        for jatom in range(iatom): 
            distances[iatom,jatom] = dist(structure[iatom,:],structure[jatom,:])
            distances[jatom,iatom] = distances[iatom,jatom]
    covalent_radii = {b'H':0.37,b'C':0.77,b'N':0.75,b'O':0.73}
    bonds = np.zeros((natom,natom))
    bond_lengths = np.zeros((natom,natom))
    # the followin loop decides wether a bond between atom iatom and jatom exists
    bond_names = []
    for iatom in range(natom):
        for jatom in range(iatom):
            ri = covalent_radii[atom_names[iatom]] 
            rj = covalent_radii[atom_names[jatom]]
            if distances[iatom,jatom] < (ri+rj)*1.3:
                bonds[iatom,jatom] = 1
                bonds[jatom,iatom] = 1
                bond_lengths[iatom,jatom] = distances[iatom,jatom]
                bond_lengths[jatom,iatom] = distances[iatom,jatom]
                bond_names.append([iatom,jatom])
    # the following calculates all the angles which iatom is the vertex 
    bond_names = np.array(bond_names)
    angle_names = []
    for iatom in range(natom):
        nbonds = np.sum(bonds[iatom,:])
        if nbonds > 1 :
            idx  = find(bonds[iatom,:],1)
            combination = list(itertools.combinations(idx,2))        
            for icomb in range(len(combination)):
                jatom = combination[icomb][0]
                katom = combination[icomb][1]
                angle_names.append([jatom,iatom,katom])
    angle_names = np.array(angle_names)
    test = []
    # the following loop finds all the possible atoms that can make adihedral, but there will be a double counting
    for iatom in range(natom):
        nbondsi = np.sum(bonds[iatom,:])
        if nbondsi > 1 : 
            indexi = find(bonds[iatom,:],1)
            for iindexi in range(len(indexi)):
                jatom = indexi[iindexi]
                nbondsj = np.sum(bonds[jatom,:])
                if nbondsj > 1 :
                    indexj = find(bonds[jatom,:],1)
                    index_iatom = find(indexj,iatom)
                    index_jatom = find(indexi,jatom)
                    indexi_prime = np.delete(indexi,index_jatom)
                    indexj_prime = np.delete(indexj,index_iatom)
                    combination = list(itertools.product(indexi_prime,indexj_prime))
                    for icomb in range(len(combination)):
                        katom = combination[icomb][0]
                        latom = combination[icomb][1]
                        test.append([katom,iatom,jatom,latom])
    test = np.array(test)
    double_count = []
    # the folowing loop finds all he double countings
    for icomb in range(len(test)) : 
        for jcomb in range(icomb) : 
            if np.array_equal(np.sort(test[icomb]),np.sort(test[jcomb])):
                double_count.append(jcomb)
    dihedral_names = np.delete(test,double_count,0)
    features = []
    for ibond in range(len(bond_names)):
        iatom = bond_names[ibond,0]
        jatom = bond_names[ibond,1]
        bond = atom_names[iatom].decode("utf-8")+str(iatom+1)+atom_names[jatom].decode("utf-8")+str(jatom+1)
        features.append(np.string_(bond))
    for iangle in range(len(angle_names)):
        iatom = angle_names[iangle,0]
        jatom = angle_names[iangle,1]
        katom = angle_names[iangle,2]
        angle = atom_names[iatom].decode("utf-8")+str(iatom+1)+atom_names[jatom].decode("utf-8")+str(jatom+1)+atom_names[katom].decode("utf-8")+str(katom+1)
        features.append(np.string_(angle))
    for idihedral in range(len(dihedral_names)):
        iatom = dihedral_names[idihedral,0]
        jatom = dihedral_names[idihedral,1]
        katom = dihedral_names[idihedral,2]
        latom = dihedral_names[idihedral,3]
        dihedral = atom_names[iatom].decode("utf-8")+str(iatom+1)+atom_names[jatom].decode("utf-8")+str(jatom+1)+atom_names[katom].decode("utf-8")+str(katom+1)+atom_names[latom].decode("utf-8")+str(latom+1)
        features.append(np.string_(dihedral))
    features = np.array(features)
    return bond_names,angle_names,dihedral_names,features

def structure_analysis(arg):
    irun = arg[0]
    istep = arg[1]
    print(irun+1,istep+1)
    structure = arg[2]
    # the following loop calculates all the distances in the structure
    bonds = np.zeros((len(bond_names),))
    for icomb in range(len(bond_names)) : 
        iatom = bond_names[icomb,0]
        jatom = bond_names[icomb,1]
        bonds[icomb] = dist(structure[iatom,:],structure[jatom,:])
    angles = np.zeros((len(angle_names),)) 
    for icomb in range(len(angle_names)) : 
        jatom = angle_names[icomb,0]
        iatom = angle_names[icomb,1]
        katom = angle_names[icomb,2]
        angles[icomb] = Tangle(structure[jatom,:],structure[iatom,:],structure[katom,:])
    dihedrals = np.zeros((len(dihedral_names),))
    # the following loop finds all the possible atoms that can make adihedral, but there will be a double counting
    for icomb in range(len(dihedral_names)) : 
        katom = dihedral_names[icomb,0]
        iatom = dihedral_names[icomb,1]
        jatom = dihedral_names[icomb,2]
        latom = dihedral_names[icomb,3]
        dihedrals[icomb] = dihedral(structure[katom,:],structure[iatom,:],structure[jatom,:],structure[latom,:])
    return np.append(bonds,np.append(angles,dihedrals))



if len(sys.argv) != 7 : 
    print("INSTRUCTION: python3.6 analysis.py nHOMO nLUMO nrun nprocessor base_address sub_address")
    exit()
base_address = sys.argv[5]
sub_address = sys.argv[6]
nrun = int(sys.argv[3])
nHOMO = int(sys.argv[1])
nLUMO = int(sys.argv[2])

# initializing the structures
# at this point I get the first xyz file and figure out what the bonds,angles,dihedrals are
fxyz = open(base_address+'/'+sub_address+'1/answer.xyz')
xyzlines = fxyz.readlines()
fxyz.close()

natom = int(xyzlines[0])
nstep = int(len(xyzlines)/(natom + 2))
atom_names = np.chararray((natom,))

for iatom in range(natom) : 
    lindex = iatom + 2
    atom_names[iatom] = np.string_(xyzlines[lindex].split()[0])
pos = np.zeros((natom,3,nstep))
for iatom in range(natom) : 
    for ix in range(3) : 
        for istep in range(nstep) :
            lidx = istep*(natom+2)+iatom+2
            pos[iatom,ix,istep] = float(xyzlines[lidx].split()[ix+1])

bond_names,angle_names,dihedral_names,features = initializer(atom_names,pos[:,:,0])
nfeature = len(features)
# At this point we have all the information initialized now we need make a big array consisting all the positions for all the 500 run cis and trans
positions = []
occupancy = []
eigen_values = []
# the following loop collectes all the positions in array positions
for irun in range(nrun) : 
    add = base_address+'/'+sub_address+str(irun+1)+"/answer.xyz"
    if not(os.path.isfile(add)) : 
        continue
    fxyz = open(add)
    xyzlines = fxyz.readlines()
    fxyz.close()
    pos_irun = np.zeros((natom,3,nstep))
    for iatom in range(natom) : 
        for ix in range(3) : 
            for istep in range(nstep) :
                lidx = istep*(natom+2)+iatom+2
                pos_irun[iatom,ix,istep] = float(xyzlines[lidx].split()[ix+1])
    positions.append(pos_irun)
 
    add = base_address+'/'+sub_address+str(irun+1)+"/occupancy_MD.dat"
    focc = open(add)
    occ_lines = focc.readlines()
    focc.close()
    occ = np.zeros((nstep,2))
    for istep in range(nstep) : 
        for i in range(2) : 
            occ[istep,0] = int(float(occ_lines[istep].split()[1]))
            occ[istep,1] = int(float(occ_lines[istep].split()[2]))
    occupancy.append(occ)

    add = base_address+'/'+sub_address+str(irun+1)+"/energies.dat"
    f_eng = open(add)
    raw_data = f_eng.read()
    mid_data = re.findall("MD\sstep\s=\s*([\d]*)\s*.*.*([-+\d\s.]*)",raw_data)
    neigen = len(mid_data[0][1].split())
    eigen = np.zeros((nstep,2))
    for istep in range(nstep):
        data = mid_data[istep][1].split()
        for ieigen in range(2):
            eigen[istep,0] = float(data[nHOMO-1])
            eigen[istep,1] = float(data[nLUMO-1])
    eigen_values.append(eigen)

arg = []
for irun in range(len(positions)):
    for istep in range(nstep):
        structure = positions[irun][:,:,istep]
        arg.append([irun,istep,structure])
ndata = len(positions)*nstep
fout = h5py.File("MY_DATA_"+str(nrun)+".hdf5","w")
dsetX = fout.create_dataset("Data",(ndata,nfeature),dtype='f')
dsetY = fout.create_dataset("Eigen_Values",(ndata,2),dtype='f')
dsetO = fout.create_dataset("Occupancy",(ndata,2),'i')
dsetF = fout.create_dataset("Features",(nfeature,),dtype='|S26')
dsetF[:] = features[:]
temp_OCC = []
temp_eigen = []
for irun in range(len(eigen_values)):
    for istep in range(nstep):
        temp_eigen.append([eigen_values[irun][istep,0],eigen_values[irun][istep,1]])
        temp_OCC.append([occupancy[irun][istep,0],occupancy[irun][istep,1]])
temp_eigen = np.array(temp_eigen)
temp_OCC = np.array(temp_OCC)
dsetY[:,:] = temp_eigen[:,:]
dsetO[:,:] = temp_OCC[:,:]
nprocess = int(sys.argv[4])

if __name__ == '__main__':
    temp_data = np.zeros((ndata,nfeature))
    p = mp.Pool(nprocess)
    results = p.map(structure_analysis,arg)
    for idata in range(ndata) : 
        irun = arg[idata][0]
        istep = arg[idata][1]
        temp_data[idata,:] = np.array(results[idata])[:]
    dsetX[:,:] = temp_data[:,:]
    
    
