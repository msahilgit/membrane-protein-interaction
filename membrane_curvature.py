import MDAnalysis as mda
import numpy as np
import copy as c
import scipy.spatial as scp


def zpositions(uni, selection, nbins=[6,6], start=0, end=None, skip=1):
    '''
    This function calculates the average z coordinate of membrane head groups
    or any surface in general (having curvature along z axis) (defined by selection):
    Basically - calculates relative z positions of any surface or membrane - indicative of curvature.
    
        > extract the z coordinates of selection
        > substrated cog
        > substracted minimum
        > a 2d (in x-y coordinates) grid defined by nbins - average z position is calculated
        > output conveys relative z positions of different parts of membrane or any surface in general.
        
    #INPUTS:
        uni       -  [mdaU] single or list of mda universes
        selection -  [atms] atom selection defining a surface - random or disjointed atoms will be meaningless
        nbins     -  [list]2 d[6,6] number of bins in x and y space
        start     -  [int]   d[0]
        end       -  [int]   d[None]
        skip      -  [int]   d[1]  start, end and skip should be defined in number of timesteps.
        
    #OUTPUTS
        zdists    -  [mat](nbin*nbin) matrix of zpositions in x-y space; unit - Angstrom
        errors    -  [mat](nbin*nbin) error in zdists
        
    '''
    
    if type(uni) == mda.core.universe.Universe:
        to_iter = [uni]
    elif any([type(uni) == i for i in [list,np.ndarray]]):
        to_iter = uni
        if all([i == mda.core.universe.Universe for i in to_iter]):
            n_atoms = [len(list(i.select_atoms(selection))) for i in to_iter]
            if not all([n_atoms[0] == i for i in n_atoms[1:]]):
                raise ValueError('Different Universe has different number of atoms \n {a}'.format(a=n_atoms))
        
    
    zdists  = np.zeros((nbins[0]+1, nbins[1]+1))
    errors  = np.zeros((nbins[0]+1, nbins[1]+1))
    counter = np.zeros((nbins[0]+1, nbins[1]+1))
    
    for uni in to_iter:
        
        if end == None:
            end = len(uni.trajectory)
        elif type(end) == int:
            end = end
        else:
            raise ValueError('problem with end')
        
        for t in tqdm(range(start, end, skip)):
            uni.trajectory[t]
            positions = uni.select_atoms(selection).positions
            zpos = positions[:,2]
            zpos -= np.mean(zpos)
            zpos -= np.min(zpos)

            xmin = np.min(positions[:,0])
            dx = ( np.max(positions[:,0]) - xmin ) / nbins[0]
            ymin = np.min(positions[:,1])
            dy = ( np.max(positions[:,1]) - ymin ) / nbins[1]
            for p in range(len(positions)):
                xbin = int(np.floor((positions[p][0] - xmin) / dx))
                ybin = int(np.floor((positions[p][1] - ymin) / dy))
                zdists[xbin][ybin] += zpos[p]
                errors[xbin][ybin] += zpos[p]**2
                counter[xbin][ybin] += 1

    zdists /= counter
    zdists[np.isnan(zdists)] = 0.0
    
    errors /= counter
    errors[np.isnan(errors)] = 0.0
    errors = np.sqrt( errors - np.square(zdists) )
    
    zdists -= np.min(zdists)
    
    return zdists, errors







def protein_location_in_membrane(uni, selection1='protein and not name H*', selection2='name P', zrange=4, 
                                 nbins=[6,6],
                                 start=0, end=None, skip=1):
    '''
    This function calculates protein location in membrane:
        to be used to estimate the protein location while studying membrane spatial protein 
        like membrane curvature (calculate by function zpositions) etc.
        WARNING: this function gives protein location relative to some selection for example membrane:
                    hence, some parameters need to be choosen carefully and in 
                    agreement with used to measure membrane surface.
                  this function assumes membrane and protein to be aligned in XY plane
                    
    #INPUTS
        uni        - [mdaU] single or list of mda universes
        selection1 - [atms] d[protein and not name H*] seletion defingin the protein or entity embedded in membrane for which location to be estimated
        selection2 - [atms] d[name P] selection defining surface, in which selection1 location to be estimated
                                        selection2 defines where the selection location is being calculated:
                                            for instance: upper or lower membrane
        zrange     - [int]  d[4] range of selection1 positions to be selected:
                                    selection1 positions within zmean(selection2 positions) +- zrange/2
        nbins      - [list] d[6,6] binning of XY surface (membrane)
        start      - [int] d[0]
        end        - [int] d[None]
        skip       - [int] d[1] start, end and skip should be defined in number of timesteps.
        
    #OUTPUTS
        zlocs      - [mat] (nbin*nbin) probability of finding selection1 in particular location (defined by bin)
    
    '''
    if type(uni) == mda.core.universe.Universe:
        to_iter = [uni]
    elif any([type(uni) == i for i in [list,np.ndarray]]):
        to_iter = uni
        if all([i == mda.core.universe.Universe for i in to_iter]):
            n_atoms = [len(list(i.select_atoms(selection1 + ' or ' + selection2))) for i in to_iter]
            if not all([n_atoms[0] == i for i in n_atoms[1:]]):
                raise ValueError('Different Universe has different number of atoms \n {a}'.format(a=n_atoms))
                
                
    zlocs = np.zeros((nbins[0]+1, nbins[1]+1))
    counter = 0
    
    for uni in to_iter:
        
        if end == None:
            end = len(uni.trajectory)
        elif type(end) == int:
            end = end
        else:
            raise ValueError('problem with end')
            
        for t in tqdm(range(start, end, skip)):
            uni.trajectory[t]
            
            positions2 = uni.select_atoms(selection2).positions
            xmin = np.min(positions2[:,0])
            dx = ( np.max(positions2[:,0]) - xmin ) / nbins[0]
            ymin = np.min(positions2[:,1])
            dy = ( np.max(positions2[:,1]) - ymin ) / nbins[1]
            zmean2 = np.mean(positions2[:,2])
            
            positions1 = uni.select_atoms(selection1).positions
            positions1 = positions1[ np.where( (positions1[:,2] >= zmean2-zrange/2) & (positions1[:,2] <= zmean2+zrange/2) ) ]
            for p in range(len(positions1)):
                xbin = int(np.floor((positions1[p][0] - xmin) / dx))
                ybin = int(np.floor((positions1[p][1] - ymin) / dy))
                zlocs[xbin][ybin] += 1
                counter += 1
            
    zlocs /= counter
    return zlocs











def calculate_nearest_member_fluctuation(uni, selection):
    '''
    This fun_function calculates the deviation of nearest member along z axis.
    To be used to calculate LIPID-FLUCTUATIONS in membrane aligned in XY plane.
    shortest distance is measured in XY plane and fluctuation along z axis.
    '''
    try:
        adiff = 0
        counter = 0
        for t in tqdm(range(len(uni.trajectory))):
            uni.trajectory[t]

            positions = uni.select_atoms(selection).positions

            if t == 0:
                print('{a} atoms are selected'.format(a=len(positions)))
                middle_point = np.mean(positions[:,2])
                upper_inds = np.where(positions[:,2] > middle_point)[0]
                lower_inds = np.where(positions[:,2] < middle_point)[0]
                if len(upper_inds)+len(lower_inds) != len(positions):
                    raise NotImplementedError('error')


            for inds in [upper_inds, lower_inds]:

                while True:
                    if len(inds) <= 1:
                        break

                    nearest = np.argmin(scp.distance.cdist([positions[:,[0,1]][inds[0]]], positions[:,[0,1]][inds[1:]]))
                    adiff += np.abs(positions[:,2][inds[0]] - positions[:,2][inds[nearest]])
                    counter += 1

                    inds = np.delete(inds,[0,nearest])

        return adiff/counter
    
    except KeyboardInterrupt:
        return 'keyboard - Interuption \t', adiff/counter


