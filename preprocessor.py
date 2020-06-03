import sys
import numpy as np
import scipy
import scipy.constants
import scipy.interpolate
import scipy.signal
from fccd import imgXraw as cleanXraw
from xcale.common.misc import printd, printv

from xcale.common.communicator import  rank
from xcale.common.communicator import  size as mpi_size
from xcale.common.communicator import  igatherv


def get_chunk_slices(n_slices):

    chunk_size =np.int( np.ceil(n_slices/mpi_size)) # ceil for better load balance
    nreduce=(chunk_size*(mpi_size)-n_slices)  # how much we overshoot the size
    start = np.concatenate((np.arange(mpi_size-nreduce)*chunk_size,
                            (mpi_size-nreduce)*chunk_size+np.arange(nreduce)*(chunk_size-1)))
    stop = np.append(start[1:],n_slices)

    start=start.reshape((mpi_size,1))
    stop=stop.reshape((mpi_size,1))
    slices=np.longlong(np.concatenate((start,stop),axis=1))
    return slices 

def get_loop_chunk_slices(ns, ms, mc ):
    # ns: num_slices, ms=mpi_size, mc=max_chunk
    # loop_chunks size
    if np.isinf(mc): 
#        print("ms",ms)
        return np.array([0,ns],dtype='int64')
#    print("glc ms",ms)

    ls=np.int(np.ceil(ns/(ms*mc)))    
    # nreduce: how many points we overshoot if we use max_chunk everywhere
    nr=ls*mc*ms-ns
    #print(nr,ls,mc)
    # number of reduced loop_chunks:
    
    cr=np.ceil(nr/ms/ls)
    # make it a multiple of 2 since we do 2 slices at once
    #cr=np.ceil(nr/ms/ls/2)*2

    if nr==0:
        rl=0
    else:
        rl=np.int(np.floor((nr/ms)/cr))
    
    loop_chunks=np.concatenate((np.arange(ls-rl)*ms*mc,(ls-rl)*ms*mc+np.arange(rl)*ms*(mc-cr),[ns]))
    return np.int64(loop_chunks)



def combine_double_exposure(data0, data1, double_exp_time_ratio, thres=3e3):

    msk=data0<thres    

    return (double_exp_time_ratio+1)*(data0*msk+data1)/(double_exp_time_ratio*msk+1)

def resolution2frame_width(final_res, detector_distance, energy, detector_pixel_size, frame_width):

    hc=scipy.constants.Planck*scipy.constants.c/scipy.constants.elementary_charge
   
    wavelength = hc/energy
    padded_frame_width = frame_width**2*detector_pixel_size*final_res/(detector_distance*wavelength)

    return padded_frame_width # cropped (TODO:or padded?) width of the raw clean frames

#Computes a weighted average of the coordinates, where if the image is stronger you have more weight.
def center_of_mass(img, coord_array_1d):
    return np.array([np.sum(img*coord_array_1d)/np.sum(img), np.sum(img*coord_array_1d.T)/np.sum(img)])


def filter_frame(frame, bbox):
    return scipy.signal.convolve2d(frame, bbox, mode='same', boundary='fill')


#Interpolation around the center of mass, thus centering. This downsamples into the output frame width
def shift_rescale(img, coord_array_1d, coord, center_of_mass):
    img_out=(scipy.interpolate.interp2d(coord_array_1d, coord_array_1d, img, fill_value=0)(coord+center_of_mass[1],coord+center_of_mass[0])).T
    img_out*=(img_out>0)
    return img_out

def split_background(background_double_exp):

    # split the average from 2 exposures:
    bkg_avg0=np.average(background_double_exp[0::2],axis=0)
    bkg_avg1=np.average(background_double_exp[1::2],axis=0)

    return np.array([bkg_avg0, bkg_avg1])


def prepare(metadata, frames, dark_frames):

    ## get one frame to compute center

    if metadata["double_exposure"]:

        background_avg = split_background(dark_frames)

        frame_exp1 = (frames[0::2] - background_avg[0])[0]
        frame_exp2 = (frames[1::2] - background_avg[1])[0]

        # get one clean frame
        clean_frame = combine_double_exposure(cleanXraw(frame_exp1), cleanXraw(frame_exp2), metadata["double_exp_time_ratio"])

    else:        
        background_avg = np.average(dark_frames,axis=0)
        clean_frame = cleanXraw(frames - background_avg)


    #TODO: When do we know this one? Is the one original shape from the beginning right?
    metadata["frame_width"] = clean_frame.shape[0]

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))

    # we need a shift, we take it from the first frame:
    com = center_of_mass(clean_frame*(clean_frame>0), xx) - metadata["frame_width"]//2
    com = np.round(com)

    metadata["center_of_mass"] = com

    #metadata["energy"]=metadata["energy"]/scipy.constants.elementary_charge
    # cropped width of the raw clean frames
    if metadata["desired_padded_input_frame_width"]:
        metadata["padded_frame_width"] = metadata["desired_padded_input_frame_width"]

    else:
        metadata["padded_frame_width"] = resolution2frame_width(metadata["final_res"], metadata["detector_distance"], metadata["energy"], metadata["detector_pixel_size"], metadata["frame_width"]) 
    
    # modify pixel size; the pixel size is rescaled
    metadata["x_pixel_size"] = metadata["detector_pixel_size"] * metadata["padded_frame_width"] / metadata["output_frame_width"]
    metadata["y_pixel_size"] = metadata["x_pixel_size"]

    #metadata['detector_pixel_size'] = metadata["x_pixel_size"]

    # frame corner
    corner_x = metadata['x_pixel_size']*metadata['output_frame_width']/2  
    corner_z = metadata['detector_distance']                
    metadata['corner_position'] = [corner_x, corner_x, corner_z]
    metadata["energy"] = metadata["energy"]*scipy.constants.elementary_charge
    return metadata, background_avg


# loop through all frames and save the result
def process_stack(metadata, frames_stack, background_avg):

    if metadata["double_exposure"]:    
        n_frames = frames_stack.shape[0]//2 # 1/2 for double exposure
    else:
        n_frames = frames_stack.shape[0]

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))

    #Convolution kernel
    kernel_width = np.max([np.int(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1])
    bbox = np.ones((kernel_width,kernel_width))

    #Coordinates of the output frame onto the grid of the input frame
    coord = np.arange(-metadata["output_frame_width"]//2, metadata["output_frame_width"]//2) / metadata["output_frame_width"] * metadata["padded_frame_width"] + metadata["frame_width"]//2

    out_data_shape = (n_frames, metadata["output_frame_width"], metadata["output_frame_width"])

    out_data = np.zeros(out_data_shape, dtype= np.float32)

    printv("\nProcessing the stack of raw frames...\n")

    for ii in np.arange(n_frames):
        
        if metadata["double_exposure"]:
            clean_frame = combine_double_exposure(cleanXraw(frames_stack[ii*2]-background_avg[0]), cleanXraw(frames_stack[ii*2+1]-background_avg[1]), metadata["double_exp_time_ratio"])

        else:
            clean_frame = cleanXraw(frames_stack[ii]-background_avg)
      
        filtered_frame = filter_frame(clean_frame, bbox)

        #Center and downsample a clean frame
        centered_rescaled_frame = shift_rescale(filtered_frame, xx, coord, metadata["center_of_mass"])


        out_data[ii] = centered_rescaled_frame
        #print('hello')
        if rank == 0 :
            sys.stdout.write('\r frame = %s/%s ' %(ii+1, n_frames))
            sys.stdout.flush()
            print("\n")

    return out_data

def calculate_mpi_chunk(n_total_frames, my_rank, n_ranks):

    frames_per_rank = n_total_frames//n_ranks

    #we always make chunks multiple of 2 because of double exposure
    if frames_per_rank % 2 == 1:
        frames_per_rank -= 1

    extra_work = 0

    if  rank == mpi_size - 1:
        extra_work =  n_total_frames - (n_ranks * frames_per_rank) 

    printv("Frames to compute per rank: " + str(frames_per_rank))

    frames_range = slice(my_rank * frames_per_rank, ((my_rank + 1) * frames_per_rank) + extra_work)

    printd("My range of ranks: " + str(frames_range) + ", my extra work: " + str(extra_work))

    return frames_range



# loop through all frames and save the result
def process_stack1(metadata, frames_stack, background_avg, out_data = None):
    #n_frames = my_indices.stop-my_indices.start
    n_frames = frames_stack.shape[0]
    #my_indices = calculate_mpi_chunk(n_frames, rank, mpi_size)
    #n_frames = my_indices.stop-my_indices.start+1
    
    if metadata["double_exposure"]:    
        n_frames //= 2 # 1/2 for double exposure
    else:
        pass
    

    #Coordinates from 0 to frame width, 1 dimension
    xx=np.reshape(np.arange(metadata["frame_width"]),(metadata["frame_width"],1))

    #Convolution kernel
    kernel_width = np.max([np.int(np.floor(metadata["padded_frame_width"]/metadata["output_frame_width"])),1])
    bbox = np.ones((kernel_width,kernel_width))

    #Coordinates of the output frame onto the grid of the input frame
    coord = np.arange(-metadata["output_frame_width"]//2, metadata["output_frame_width"]//2) / metadata["output_frame_width"] * metadata["padded_frame_width"] + metadata["frame_width"]//2

    #if type(out_data) == type(None):
    #    out_data_shape = (n_frames, metadata["output_frame_width"], metadata["output_frame_width"])
    #    out_data = np.zeros(out_data_shape, dtype= np.float32)

    if metadata['double_exposure']:
        printv("\nProcessing the stack of raw frames - double exposure...\n")
    else:
        printv("\nProcessing the stack of raw frames...\n")


    max_chunk_slice = 1
    loop_chunks=get_loop_chunk_slices(n_frames, mpi_size, max_chunk_slice )

    frames_local=None
    pgather = None
    if rank == 0:
        out_data_shape = (max_chunk_slice*mpi_size, metadata["output_frame_width"], metadata["output_frame_width"])        
        frames_chunks = np.empty(out_data_shape,dtype=np.float32)
        
    for ii in range(loop_chunks.size-1):
    #for ii in range(loop_chunks.size-2,loop_chunks.size-1):
        nslices = loop_chunks[ii+1]-loop_chunks[ii]
        chunk_slices = get_chunk_slices(nslices) 
        chunks=chunk_slices[rank,:]+loop_chunks[ii]
        
        
        #printv( 'loop_chunk {}/{}:{}, mpi chunks {}'.format(ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
        #printv( 'chunks {}'.format(chunks))
        
        #print('rank',rank,'chunks',chunks[1]-chunks[0])
        
        # only one frame per chunk
        ii_frames= chunks[0]
        
        # empty 
        if chunks[1]-chunks[0] == 0:
            centered_rescaled_frame = np.empty((0),dtype = np.float32)
        else:            
        
            if metadata["double_exposure"]:
                clean_frame = combine_double_exposure(cleanXraw(frames_stack[ii_frames*2]-background_avg[0]), cleanXraw(frames_stack[ii_frames*2+1]-background_avg[1]), metadata["double_exp_time_ratio"])
                
            else:
                clean_frame = cleanXraw(frames_stack[ii_frames]-background_avg)
          
            filtered_frame = filter_frame(clean_frame, bbox)
    
            #Center and downsample a clean frame
            centered_rescaled_frame = shift_rescale(filtered_frame, xx, coord, metadata["center_of_mass"])
            centered_rescaled_frame = np.float32(centered_rescaled_frame)
    
        
            stack_shape = (1,centered_rescaled_frame.shape[0], centered_rescaled_frame.shape[1])
    
            
            centered_rescaled_frame =np.reshape(centered_rescaled_frame, stack_shape) 
        
        if rank ==0:
            frames_local =  frames_chunks[0:loop_chunks[ii+1]-loop_chunks[ii],:,:]


        
        pgather = igatherv(centered_rescaled_frame,chunk_slices,data=frames_local)   
        
        if mpi_size > 1:
            pgather.Wait()

        if rank == 0:
            out_data[loop_chunks[ii]:loop_chunks[ii+1],:,:] = frames_local
                   
        
        #print('hello')
        if rank == 0 :
            #out_data.flush()
            ii_rframe = ii_frames#*(metadata["double_exposure"]+1) 
            sys.stdout.write('\r frame {}/{}, loop_chunk {}/{}:{}, mpi chunks {}'.format(ii_rframe+1, n_frames, ii+1,loop_chunks.size-1, loop_chunks[ii:ii+2],loop_chunks[ii]+np.append(chunk_slices[:,0],chunk_slices[-1,1]).ravel()))
#            sys.stdout.write('\r frame = %s/%s ' %(ii_frames+1, n_frames))
            sys.stdout.flush()
            #print("\n")


    if rank == 0:
        out_data.flush()
        print("\n")

    return out_data

        