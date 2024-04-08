import numpy as np

def isint(input):
    if np.round(input) == input:
        return True
    else:
        return False
        
def isodd(input):
    if np.mod(input,2) == 1:
        return True
    elif np.mod(input,2) == 0:
        return False
    else:
        print('isodd function error: the input variable is not a integer.')
        return -1
    
def iseven(input):
    if np.mod(input,2) == 0:
        return True
    elif np.mod(input,2) == 1:
        return False
    else:
        print('iseven function error: the input variable is not a integer.')
        return -1
   
def matrix_clip(matrix_in,row_cen,col_cen,clip_size):
    matrix = np.copy(matrix_in)
    # matrix is a nd array
    # row_cne and col_cen must be integers
    # clip_size must be a odd number
    clip_size = np.round(clip_size)
    if iseven(clip_size):
        clip_size = clip_size + 1
    
    one_side_size = (clip_size - 1)/2
    
    row_start   = int(row_cen - one_side_size)
    row_end     = int(row_cen + one_side_size)
    col_start   = int(col_cen - one_side_size)
    col_end     = int(col_cen + one_side_size)
    
    # check the clip area over the area of matrix or not and the dimensions of the matrix
    if row_start < 0 or col_start < 0:
        print('tools.matrix_clip: clip area over the area of the input matrix.')
        return -1
    
    if matrix.ndim == 2:
        [matrix_row_size, matrix_col_size] = matrix.shape
        if row_end > matrix_row_size-1 or col_end > matrix_col_size-1:
            print('tools.matrix_clip: clip area over the area of the input matrix.')
            return -1       
    elif matrix.ndim == 3:
        [matrix_frame_size,matrix_row_size, matrix_col_size] = matrix.shape
        if row_end > matrix_row_size-1 or col_end > matrix_col_size-1:
            print('tools.matrix_clip: clip area over the input matrix.')
            return -1
    else:
        print('tools.matrix_clip: input matrix should be a 2D or 3D ndarray.')
        return -1
        
    # clip the interesting part
    if matrix.ndim == 2:
        matrix_output = matrix[row_start:row_end+1,col_start:col_end+1]
    elif matrix.ndim ==3:
        matrix_output = matrix[:,row_start:row_end+1,col_start:col_end+1]
        
    return matrix_output

def cal_real_space_pixel_res(wavelength = None, detector_distance = None, clip_size = None, pixel_size = None):
    real_space_pixel_res = wavelength * detector_distance / clip_size / pixel_size
    return real_space_pixel_res
        