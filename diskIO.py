import numpy as np
import sys
import os
import json
from PIL import Image
import xcale.common as ptycommon
from xcale.common.misc import printd, printv


def read_metadata(json_file):

    with open(json_file) as f:
        metadata = json.load(f)

    if "translations" in metadata:

        metadata["translations"] = np.array(metadata["translations"])

    return metadata


def read_dark_data(metadata, json_file):


    dark_frames = None

    base_folder = os.path.split(json_file)[:-1][0] + "/"

    if "dark_dir" in metadata:

        printv("\nReading dark frames from disk...\n")

        #by default we could take the full path, but in this case it has an absolute path from PHASIS,
        #which is not good if you move data to other places
        dark_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["dark_dir"])))


    return dark_frames



def read_frames(metadata, json_file, my_indexes):

    raw_frames = None

    base_folder = os.path.split(json_file)[:-1][0] + "/"
    directory = base_folder + os.path.basename(os.path.normpath(metadata["exp_dir"]))
    
    printv("\nReading raw frames from disk...\n")
    #raw_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["exp_dir"])), my_indexes)
    raw_frames = read_tiffs(directory, my_indexes)

    return raw_frames

def read_data(metadata, json_file, my_indexes):


    dark_frames = None
    raw_frames = None

    base_folder = os.path.split(json_file)[:-1][0] + "/"

    if "dark_dir" in metadata:

        printv("\nReading dark frames from disk...\n")

        #by default we could take the full path, but in this case it has an absolute path from PHASIS,
        #which is not good if you move data to other places
        dark_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["dark_dir"])))

    if "exp_dir" in metadata:
        
        printv("\nReading raw frames from disk...\n")

        raw_frames = read_tiffs(base_folder + os.path.basename(os.path.normpath(metadata["exp_dir"])), my_indexes)

    return dark_frames, raw_frames


def read_tiffs(directory, my_indexes = None):
    from xcale.common.communicator import rank

    lst=os.listdir(directory)
    lst.sort()

    #We remove from the list anything that is not a tiff
    lst = [item for item in lst if item.endswith('.tif')]

    if my_indexes is not None:
        lst = lst[my_indexes]

    frames = []    

    ii=0
    n_frames=len(lst)

    for fname in lst:       
        ii+=1
        im = Image.open(os.path.join(directory, fname))
        imarray = np.array(im)
        frames.append(imarray)
        if rank == 0:
            sys.stdout.write('\r file = %s/%s ' %(ii,n_frames))
            sys.stdout.flush()
        
    if rank == 0: print("\n")
    return np.array(frames)

def frames_out(file_name, shape_frames):
    import h5py
    #from xcale.common.communicator import  rank, mpi_barrier, comm
 
    #if data_format is None: 
    #    data_format = self.metadataFormat 
    
    #fid = h5py.File(file_name, 'a', driver='mpio', comm=comm)
    fid = h5py.File(file_name, 'a')
        
    if not "entry_1/data_1/" in fid: 
        fid.create_group("entry_1/data_1")
    if not '/entry_1/instrument_1/detector_1/' in fid:
        fid.create_group('/entry_1/instrument_1/detector_1/')

    #out_frames = fid.create_dataset('entry_1/data_1/data', shape_frames , dtype='float32')
    out_frames = fid.create_dataset('/entry_1/instrument_1/detector_1/data', shape_frames , dtype='float32')
    

    if "entry_1/instrument_1/detector_1/data" in fid and not "entry_1/data_1/data" in fid:
        fid["entry_1/instrument_1/detector_1/data"].attrs['axes'] = "translation:y:x" 
        fid["entry_1/data_1/data"] = h5py.SoftLink("/entry_1/instrument_1/detector_1/data")

            

    return out_frames, fid

def map_tiffs(base_folder):
    import tifffile
    ## tifs = tifffile.TiffSequence(lst)
    ##tifs = tifffile.TiffSequence(base_folder)
    tifs = tifffile.TiffSequence(base_folder+'/*.tif')
    class MyClass():
        def __getitem__(self, key):
            return np.array(tifs.asarray(int(key)))
        shape=tifs.shape

    myobj = MyClass()
    
    return myobj
 
