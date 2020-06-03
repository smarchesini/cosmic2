import h5py
import numpy as np
import sys


if __name__ == '__main__':

    args = sys.argv[1:]


    file_1 = args[0]

    file_2 = args[1]

    fid_1 = h5py.File(file_1, 'r')
    fid_2 = h5py.File(file_2, 'r')

    A = fid_1["entry_1/data_1/data"][...]
    B = fid_2["entry_1/data_1/data"][...]  
    print(A)
    print(B)
    print(np.array_equal(A, B))

    print("Sum: " + str(np.sum(A-B)))

    fid_1.close()
    fid_2.close()

