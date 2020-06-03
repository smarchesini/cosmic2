import numpy as np
import xcale
import xcale.common as ptycommon
from xcale.common.cupy_common import memcopy_to_device, memcopy_to_host
from xcale.common.misc import printd, printv
from xcale.common.cupy_common import check_cupy_available
from xcale.common.communicator import mpi_allGather, mpi_Bcast, mpi_Gather, rank, size, mpi_barrier
import sys
import os
import diskIO

import preprocessor

from diskIO import frames_out, map_tiffs


def convert_translations(translations):

    zeros = np.zeros(translations.shape[0])

    new_translations = np.column_stack((translations[:,1]*1e-6,translations[::-1,0]*1e-6, zeros)) 

    return new_translations

if True:
    json_file = '/fastdata/NS/Fe/200315003/200315003_002_info.json'
 
if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        json_file = args[0]
    except:
        pass

    options = {"debug": True,
               "gpu_accelerated": False}

    gpu_available = check_cupy_available()

    if not ptycommon.communicator.mpi_enabled:
        printd("\nWARNING: mpi4py is not installed. MPI communication and partition won't be performed.\n" + \
               "Verify mpi4py is properly installed if you want to enable MPI communication.\n")

    if not gpu_available and options["gpu_accelerated"]:
        printd("\nWARNING: GPU mode was selected, but CuPy is not installed.\n" + \
               "Verify CuPy is properly installed to enable GPU acceleration.\n" + \
               "Switching to CPU mode...\n")

        options["gpu_accelerated"] = False


 

    metadata = diskIO.read_metadata(json_file)

    #These do not change
    metadata["detector_pixel_size"] = 30e-6  # 30 microns
    metadata["detector_distance"] = 0.121 #this is in meters

    #These here below are options, we can edit them
    metadata["final_res"] = 5e-9 # desired final pixel size nanometers
    metadata["desired_padded_input_frame_width"] = None
    metadata["output_frame_width"] = 256 # final frame width 
    metadata["translations"] = convert_translations(metadata["translations"])
    #Energy from eV to Joules
    #import scipy.constants
    #metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge



    n_total_frames = metadata["translations"].shape[0]
    if metadata["double_exposure"]: n_total_frames *= 2

    dark_frames = diskIO.read_dark_data(metadata, json_file)

    ##########   

    base_folder = os.path.split(json_file)[:-1][0] + "/" 
    base_folder += os.path.basename(os.path.normpath(metadata["exp_dir"]))
  
    ##########
    raw_frames = map_tiffs(base_folder)

    n_frames = raw_frames.shape[0]

    #This takes the center of a chunk as a center frame(s)
    #center = np.int(n_frames)//2
    #This takes the center of the stack as a center frame(s)
    center = np.int(n_total_frames)//2

    #Check this in case we are in double exposure
    if center % 2 == 1:
        center -= 1

    if metadata["double_exposure"]:
        metadata["double_exp_time_ratio"] = metadata["dwell1"] // metadata["dwell2"] # time ratio between long and short exposure
        center_frames = np.array([raw_frames[center], raw_frames[center + 1]])
    else:
        center_frames = raw_frames[center]
    
    
    #print('energy (eV)',metadata['energy'])
    metadata, background_avg =  preprocessor.prepare(metadata, center_frames, dark_frames)
    #print('energy (J)',metadata['energy'])
    #we take the center of mass from rank 0
    #metadata["center_of_mass"] = mpi_Bcast(metadata["center_of_mass"], metadata["center_of_mass"], 0, mode = "cpu")


    io = ptycommon.IO()
    output_filename = os.path.splitext(json_file)[:-1][0][:-4] + "cosmic2.cxi"
    
    if rank == 0:

        #output_filename = os.path.splitext(json_file)[:-1][0] + "_cosmic2.cxi"

        printv("\nSaving cxi file metadata: " + output_filename + "\n")

        #data_dictionary["data"] = np.concatenate(data_dictionary["data"], axis=0)

        #This script deletes and rewrites a previous file with the same name
        try:
            os.remove(output_filename)
        except OSError:
            pass

        #import time
        #print('hello')
        #time.sleep(10)

        io.write(output_filename, metadata, data_format = io.metadataFormat) #We generate a new cxi with the new data
        #io.write(output_filename, data_dictionary, data_format = io.dataFormat) #We use the metadata we read above and drop it into the new cxi


    data_shape = (n_frames//(metadata['double_exposure']+1), metadata["output_frame_width"], metadata["output_frame_width"])
    if rank == 0:
        out_frames, fid = frames_out(output_filename, data_shape)  
    else:
        out_frames = 0
    
    #printv('processing stacks')    
    #my_indexes = calculate_mpi_chunk(n_total_frames, rank, size)

    output_data = preprocessor.process_stack1(metadata,raw_frames,background_avg, out_frames)
    if rank ==0:
        fid.close()
    
    #output_data = preprocessor.process_stack(metadata, raw_frames, background_avg)

    #data_dictionary = {}
    #data_dictionary.update({"data" : output_data})


    #we gather all results into rank 0
    #data_dictionary["data"] = mpi_Gather(data_dictionary["data"], data_dictionary["data"], 0, mode = "cpu")

