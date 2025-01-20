import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

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
    
def frame_binning(data = None,binning_factor = None):
    if data.ndim == 2:
        [row_size,col_size] = data.shape
        row_range = (row_size//binning_factor)*binning_factor
        col_rnage = (col_size//binning_factor)*binning_factor
        binning_data = np.copy(data[0:row_range,0:col_rnage]) # clip data to fit the binning factor
        
        binning_data = binning_data.reshape(row_size // binning_factor, binning_factor, col_size // binning_factor, binning_factor).sum(axis=(1, 3))
        
    elif data.ndim == 3:
        [frame_num,row_size,col_size] = data.shape
        row_range = (row_size//binning_factor)*binning_factor
        col_rnage = (col_size//binning_factor)*binning_factor
        binning_data = np.copy(data[:,0:row_range,0:col_rnage]) # clip data to fit the binning factor
        
        binning_data = binning_data.reshape(frame_num,row_size // binning_factor, binning_factor, col_size // binning_factor, binning_factor).sum(axis=(2, 4))
        
    if data.dtype == 'bool':
        binning_data = binning_data != 0
        
    return binning_data

def frame_resampling(ori_frame=None, resampling_factor=None, interpolation="linear"):
    """
    Resample a 3D image, only resizing the height and width (not depth).

    Parameters:
    ori_frame (numpy.ndarray): Input 3D image (shape: [depth, height, width]).
    resampling_factor (float or tuple): Scaling factor for height & width.
    interpolation (str): Interpolation method, can be "linear", "nearest", or "cubic".

    Returns:
    numpy.ndarray: Resampled 3D image.
    """
    if ori_frame is None or resampling_factor is None:
        raise ValueError("Both 'ori_frame' and 'resampling_factor' must be provided.")

    if ori_frame.ndim != 3:
        raise ValueError("Input must be a 3D image with shape (depth, height, width).")

    # 解析 resampling_factor
    if isinstance(resampling_factor, (int, float)):
        resampling_factor = (resampling_factor, resampling_factor)  # 统一缩放 height & width
    elif isinstance(resampling_factor, (list, tuple)) and len(resampling_factor) == 2:
        resampling_factor = tuple(resampling_factor)
    else:
        raise ValueError("resampling_factor must be a float or a tuple (height_factor, width_factor).")

    # 选择插值方法
    interp_order = {"nearest": 0, "linear": 1, "cubic": 3}
    if interpolation not in interp_order:
        raise ValueError("Interpolation must be 'nearest', 'linear', or 'cubic'.")

    # 计算缩放比例 (depth 维度保持不变)
    resize_factors = (1,) + resampling_factor  # (depth_scale=1, height_scale, width_scale)

    # 处理不同数据类型
    if ori_frame.dtype == np.bool_:  # 处理布尔类型
        mask = (~ori_frame).astype(np.float32)
        mask = zoom(mask, resize_factors, order=interp_order[interpolation])
        resize_frame = ~(mask > 0)

    elif np.iscomplexobj(ori_frame):  # 处理复数类型
        amp = zoom(np.abs(ori_frame), resize_factors, order=interp_order[interpolation])
        phase = zoom(np.angle(ori_frame), resize_factors, order=interp_order[interpolation])
        resize_frame = amp * np.exp(1j * phase)

    else:  # 处理普通数值类型
        resize_frame = zoom(ori_frame, resize_factors, order=interp_order[interpolation])

    return resize_frame
    