import numpy as np
import cv2
from matplotlib import pyplot as plt

def array_fft(array_in):
    array_out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array_in)))
    return array_out

def array_ifft(array_in):
    array_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array_in)))
    return array_out

def isnone(input):
    if input is None:
        return True
    else:
        return False

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

def show_length_with_unit(length,precision = 3):
    order = np.log10(np.abs(length)).astype(int)-1
    if order > 0:
        output = '{{:.{}f}}'.format(precision).format(length) + ' [m]'
    if order>=-3 and order < 0:
        output =  '{{:.{}f}}'.format(precision).format(length*1E3) + ' [mm]'
    if order>=-6 and order < -3:
        output =  '{{:.{}f}}'.format(precision).format(length*1E6) + ' [um]'
    if order>=-9 and order < -6:
        output =  '{{:.{}f}}'.format(precision).format(length*1E9) + ' [nm]'
    if order>=-12 and order < -9:
        output =  '{{:.{}f}}'.format(precision).format(length*1E12) + ' [pm]'
    if order>=-15 and order < -12:
        output =  '{{:.{}f}}'.format(precision).format(length*1E15) + ' [fm]'
    
    return output

def position_to_index(pos_x = None, pos_z=None ,x_axis = None,z_axis = None):
    if type(pos_x) is float:
        pos_x = np.array([pos_x])
    if type(pos_z) is float:
        pos_z = np.array([pos_z])
             
    row_idx = np.argmin(abs(z_axis-pos_z[:,np.newaxis]),axis=1)
    col_idx = np.argmin(abs(x_axis-pos_x[:,np.newaxis]),axis=1)
    
    return row_idx, col_idx

def energy_eV_to_wavelength_m(energy = None):
    h=6.62607015*10**-34
    e=1.60217651019*10**-19
    c=299792458.0
    wavelength=(h*c)/(energy*e)
    
    # equal to
    # h = 4.13566727E-15
    # c = 299792458.0
    # wavelength = (h*c)/energy
    
    return wavelength

def wavelength_m_to_energy_eV(wavelength = None):
    h=6.62607015*10**-34
    e=1.60217651019*10**-19
    c=299792458.0
    energy=(h*c)/(wavelength*e)
    
    # equal to
    # h = 4.13566727E-15
    # c = 299792458.0
    # energy = (h*c)/wavelength
    
    return energy

def frame_resampling(ori_frame=None,resampling_factor=None):
    # cv2.resize(ori_frame,(new_col_size,new_row_size),interpolation=cv2.INTER_LINEAR)
    if ori_frame.ndim == 2:
        ori_row_size,ori_col_size = ori_frame.shape
    elif ori_frame.ndim == 3:
        print('img_resize only can apply on a 2D image.')
        return
    
    resize_row_size = np.int32(np.round(ori_row_size*resampling_factor))
    resize_col_size = np.int32(np.round(ori_col_size*resampling_factor))
    
    if np.mod(resize_row_size,2) == 0:
        resize_row_size = resize_row_size-1
    if np.mod(resize_col_size,2) == 0:
        resize_col_size = resize_col_size-1    
        
    if ori_frame.dtype == 'bool': # for roi
        mask = (~ori_frame).astype(float)
        mask = cv2.resize(mask,(resize_col_size,resize_row_size),interpolation=cv2.INTER_LINEAR)
        resize_frame = ~(mask>0)
    elif ori_frame.dtype == 'complex128':  # for complex image
        amp   = cv2.resize(np.abs(ori_frame)  ,(resize_col_size,resize_row_size),interpolation=cv2.INTER_LINEAR)
        phase = cv2.resize(np.angle(ori_frame),(resize_col_size,resize_row_size),interpolation=cv2.INTER_LINEAR)
        resize_frame = amp*np.exp(1j*phase)
    else:
        resize_frame = cv2.resize(ori_img,(resize_col_size,resize_row_size),interpolation=cv2.INTER_LINEAR)        
    
    return resize_frame

def frame_clip(ori_frame = None, clip_row_cen = None, clip_col_cen = None ,clip_row_size = None, clip_col_size = None):
    # clip a area with row_size and col_size from the center of clip_row_idx and clip_col_idx
    if iseven(clip_row_size) or iseven(clip_col_size):
        print('Error: Clip size must be in odd.')
    
    ori_frame_num, ori_frame_row_size, ori_frame_col_size = ori_frame.shape
    extend_row_pixel = np.int32((clip_row_size-1)/2)
    extend_col_pixel = np.int32((clip_col_size-1)/2)
    clip_frame_row_start = clip_row_cen - extend_row_pixel
    clip_frame_row_end = clip_row_cen + extend_row_pixel
    clip_frame_col_start = clip_col_cen - extend_col_pixel
    clip_frame_col_end = clip_col_cen + extend_col_pixel
    
    output_frame = ori_frame[:,clip_frame_row_start:clip_frame_row_end+1,clip_frame_col_start:clip_frame_col_end+1]
    
    return output_frame

def frame_central_clip(ori_frame = None, clip_row_size = None, clip_col_size = None):
    # clip a square area with clip_size from the center of the frame
    ori_frame_num, ori_frame_row_size, ori_frame_col_size = ori_frame.shape
    if iseven(ori_frame_row_size)  or iseven(ori_frame_row_size):
        print('Error: Input frames for frame_central_clip must be odds in row and col.')
        return
    if iseven(clip_row_size) or iseven(clip_col_size):
        print('Error: Clip size must be in odd.')
        return
        
    ori_frame_row_cen = np.int32((ori_frame_row_size-1)/2)
    ori_frame_col_cen = np.int32((ori_frame_col_size-1)/2)
    
    output_frame = frame_clip(ori_frame = ori_frame, clip_row_cen = ori_frame_row_cen, clip_col_cen = ori_frame_col_cen ,clip_row_size = clip_row_size, clip_col_size = clip_col_size )
    
    return output_frame

def plot_wavefield(wavefield_object = None,frame = None):
    data = wavefield_object.data[frame]
    extent=[wavefield_object.x_axis[0],wavefield_object.x_axis[-1],wavefield_object.z_axis[0],wavefield_object.z_axis[-1]] 
    plt.figure(1)
    plt.imshow(np.abs(data),extent=extent)
    plt.figure(2)
    plt.imshow(np.angle(data),extent=extent)
    plt.show()
    