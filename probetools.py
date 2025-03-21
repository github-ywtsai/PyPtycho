import numpy as np
from numpy import pi
import tools
import material_property
import matplotlib.pyplot as plt


class wavefield_object:
    def __init__(self):
        self.attribute = 'wavefield'
        self.data                   = None
        self.x_axis                 = None
        self.z_axis                 = None
        self.pixel_res              = None
        self.wavelength             = None
        self.energy                 = None
        
class zoneplate_object(wavefield_object):
    def __init__(self):
        super().__init__() # adapt the father structure
        self.attribute = 'zoneplate'
        self.focal_length           = None
        

def gen_zoneplate(zoneplate_config):
    # energy in eV
    # dr: the min. pitch
    # N: the period for the zone plate design
    # using formula from Attwood Soft X-rays and Extereme Ultraviolet Radiation
    # material: zoneplate made of
    # thickness: thickness of the zoneplate [m]
    
    # 96 um zoneplate in TPS 25A
    # dr = 30e-9
    # N = 800
    # thickness = 600E-9
    
    
    # 60 um zoneplate in TPS 25A
    # dr = 50e-9
    # N = 300
    # thickness = 600E-9
    
    # 80 um zoneplate in TPS 25A by ANT
    # dr = 70e-9 
    # N = 286
    # thickness = 1500E-9
    # Ex:
    # zp = probetools.gen_zoneplate(dr=70e-9,N=286,energy=8979,material='Au',thickness=1500E-9)
    
    # 120 um zoneplate in TPS 25A by XRnanotech
    # dr = 70e-9
    # N = 429
    # thickness = 1500E-9
    dr          = zoneplate_config.dr
    N           = zoneplate_config.N
    energy      = zoneplate_config.energy
    material    = zoneplate_config.material
    thickness   = zoneplate_config.thickness
    
    print('Generating wavefield of the zone plate.')
    wavelength = tools.energy_eV_to_wavelength_m(energy)
    material_properties = material_property.atomic_database(material)
    n_refractivity = material_properties.calculate_refractive_index(energy = energy,theta = 0)
    
    # diameter and focal length of the zone plate
    D = 4*N*dr
    f = (4*N*dr**2)/wavelength
    ## temporary 

    # E(z) = E0 * exp(i * n k z)
    # modulator = E(z)/E0 = exp(i * n k z)
    zp_modulation_factor = np.exp(1j * n_refractivity * 2*pi/wavelength * thickness)/np.exp(1j * 1 * 2*pi/wavelength * thickness)

    # create zp
    
    n = np.arange(N+1) # n = 0,1,2,3,....,N
    rn = np.sqrt(n*wavelength*f+(n*wavelength)**2/4)
    
    pix_res = dr/2
    extend_range = D*2
    pix_num = np.int32(np.round(extend_range/pix_res))
    pix_num = tools.if_even_to_odd(pix_num, method = 1)
    cen_idx = (pix_num-1)/2
    zp = np.ones([pix_num,pix_num],dtype = np.complex128)
    
    x_axis = (np.arange(pix_num)-cen_idx)*pix_res
    y_axis = x_axis*-1 # the direction of y is inverse of the row axis
    x_matrix,y_matrix = np.meshgrid(x_axis,y_axis)
    distance_map = np.sqrt(x_matrix**2 + y_matrix**2)
    
    for n_sn in n:
        if np.mod(n_sn,2)!=0:
            mask = (distance_map <= rn[n_sn]) & (distance_map > rn[n_sn-1])
            zp[mask] = zp_modulation_factor

    
    # cs part
    cs_diameter     = zoneplate_config.cs_diameter
    cs_material     = zoneplate_config.cs_material
    cs_thickness    = zoneplate_config.cs_thickness
    if cs_diameter == None or cs_material == None or cs_thickness == None:
        print('Central stop disable.')
    else:
        wavelength = tools.energy_eV_to_wavelength_m(energy)
        property = material_property.atomic_database(cs_material)
        n_refractivity = property.calculate_refractive_index(energy = energy,theta = 0)
        cs_modulation_factor = np.exp(1j * n_refractivity * 2*pi/wavelength * cs_thickness)/np.exp(1j * 1 * 2*pi/wavelength * cs_thickness)
        cs = np.ones([pix_num,pix_num],dtype = np.complex128)
        matrix_x, matrix_y = np.meshgrid(x_axis,y_axis)
        distance_map = np.sqrt(matrix_x**2+matrix_y**2)
        cs[distance_map<cs_diameter/2] = cs_modulation_factor
        zp = zp*cs
            
    zp_object = zoneplate_object()
    zp_object.data = zp.reshape(1,zp.shape[0],zp.shape[1])
    zp_object.x_axis = x_axis
    zp_object.z_axis = x_axis
    zp_object.pixel_res = pix_res
    zp_object.focal_length = f
    zp_object.wavelength = wavelength
    zp_object.energy = energy
    
    return zp_object


def wavefield_propagating(origin_wavefield_object = None, z = None, window = None):
    # the xi_axis and eta_axis are the x_axis and y_axis of the input wavefield_object
    # wavelength in meter
    # z and propagating direction:
    # in formula, +z means propagating to "downstream" and -z to
    # "upstream". So, when input, +z means "downstream" and to increase the distance from the
    # source coming from the undulator. -z means "upstream" and to decrease
    # probe should be a ndarray
    
    # check the dimension of the input probe ndarray
    # probe must be a 3-d matrix (multi-probe)
    # when a 2-d probe input, rearange it.
    print('Calculating wavefield propagating.')
    wavefield = origin_wavefield_object.data
    wavelength = origin_wavefield_object.wavelength
    energy = origin_wavefield_object.energy
    pixel_res = origin_wavefield_object.pixel_res
    xi_axis = origin_wavefield_object.x_axis
    eta_axis = origin_wavefield_object.z_axis
    
    if wavefield.ndim == 2: # a 2D case, repackage to a 1 frame 3D case
        print('Error: wavefield_propagating')
        print('Wavefield must be a 3D array (multi-frame).')
        wavefield = wavefield.reshape(1,wavefield.shape[0],wavefield.shape[1])
    
    k = 2*pi/wavelength
    
    xi_axis_res = pixel_res
    eta_axis_res = pixel_res
    
    xi,eta = np.meshgrid(xi_axis,eta_axis) #[m]
    U_measured_eta_size,U_measured_xi_size = wavefield[0].shape
    xp_res = np.abs(wavelength*z/U_measured_xi_size/xi_axis_res)
    yp_res = np.abs(wavelength*z/U_measured_eta_size/eta_axis_res)
    U_propagated_yp_size,U_propagated_xp_size = wavefield[0].shape
    xp_axis = (np.arange(U_propagated_xp_size)-(U_propagated_xp_size-1)/2)*xp_res
    yp_axis = (np.arange(U_propagated_yp_size)-(U_propagated_yp_size-1)/2)*yp_res
    xp,yp = np.meshgrid(xp_axis,yp_axis)
    
    if z<0:
        U_measured = np.rot90(wavefield,axes=(1,2),k=2)
    else:
        U_measured = wavefield
    
    
    # calculate propagating
    temp = U_measured*np.exp(1j*k/(2*z)*(xi**2 +eta**2))
    fft_temp = tools.array_fft(temp)*xi_axis_res*eta_axis_res
    U_propagated = np.exp(1j*k*z)/(1j*wavelength*z)*np.exp(1j*k/(2*z)*(xp**2+yp**2))*fft_temp
    
    propagated_wavefield = U_propagated
    propagated_x_axis = xp_axis
    propagated_y_axis = yp_axis
    
    if window is not None:
        accept_num_pixel = np.int32(np.round(window/xp_res))
        if np.mod(accept_num_pixel,2)==0:
            accept_num_pixel += 1
        __, row_size, col_size = propagated_wavefield.shape
        row_cen = np.int32(np.round((row_size - 1)/2))
        col_cen = np.int32(np.round((col_size - 1)/2))
        extend_range = np.int32((accept_num_pixel-1)/2)
        row_start = row_cen - extend_range
        row_end = row_start + accept_num_pixel
        col_start = col_cen - extend_range
        col_end = col_start + accept_num_pixel
      
        propagated_wavefield = propagated_wavefield[:,row_start:row_end,col_start:col_end]
        propagated_x_axis = propagated_x_axis[col_start:col_end]
        propagated_y_axis = propagated_y_axis[row_start:row_end]
        
    
    #if propagated_wavefield.shape[0] == 1: # a 2D case
    #    propagated_wavefield = propagated_wavefield[0]
        
    propagated_wavefield_object = wavefield_object()
    propagated_wavefield_object.data                   = propagated_wavefield
    propagated_wavefield_object.x_axis                 = propagated_x_axis
    propagated_wavefield_object.z_axis                 = propagated_y_axis
    propagated_wavefield_object.pixel_res              = xp_res
    propagated_wavefield_object.wavelength             = wavelength
    propagated_wavefield_object.energy                 = energy
    
    return propagated_wavefield_object
    
    
def wavefield_matching(reference_wavfield_object = None, target_wavefield_object = None):
    # modified the target_wavefield_object to match the configuration of the reference_wavfield_object
    # including size, resolution, etc.
    resampling_factor = 1/(reference_wavfield_object.pixel_res/target_wavefield_object.pixel_res)
    
    # rough cut the FOV similar to the reference
    rough_window = (np.max(reference_wavfield_object.x_axis) -  np.min(reference_wavfield_object.x_axis))*1.1
    rough_clip_size = np.int32(rough_window/reference_wavfield_object.pixel_res)
    rough_clip_size = tools.if_even_to_odd(rough_clip_size, method=1)
    
    reduced_target_frame = tools.frame_central_clip(ori_frame = target_wavefield_object.data, clip_row_size = rough_clip_size, clip_col_size = rough_clip_size)
    
    target_frame_num, __, __ = target_wavefield_object.data.shape
    resampling_frame = np.empty((target_frame_num,), dtype=object) # create a new array contains target_frame_num frames
    for frame_sn in range(target_frame_num):
        temp = tools.frame_resampling(ori_frame=reduced_target_frame[frame_sn],resampling_factor=resampling_factor)
        row_size, col_size = temp.shape
        if tools.iseven(row_size):
            temp = temp[0:-1,:]
        if tools.iseven(col_size):
            temp = temp[:,0:-1]
        resampling_frame[frame_sn] = temp
    
    resampling_frame = np.stack(resampling_frame, axis=0) # reshape the array
    
    # cut interesting part    
    reference_frame_num, reference_frame_row_size, reference_frame_col_size = reference_wavfield_object.data.shape
    
    output_frame = tools.frame_central_clip(ori_frame = resampling_frame, clip_row_size = reference_frame_row_size, clip_col_size = reference_frame_col_size)
    
    output_wavefield_object = wavefield_object()
    output_wavefield_object.data                     = output_frame
    output_wavefield_object.x_axis                   = reference_wavfield_object.x_axis
    output_wavefield_object.z_axis                   = reference_wavfield_object.z_axis
    output_wavefield_object.pixel_res                = reference_wavfield_object.pixel_res
    output_wavefield_object.wavelength               = reference_wavfield_object.wavelength
    output_wavefield_object.energy                   = reference_wavfield_object.energy
    
    return output_wavefield_object