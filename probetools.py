import numpy as np
from numpy import pi
import tools
import material_property


class wavefield_object():
    def __init__(self):
        self.data                   = None
        self.x_axis                 = None
        self.z_axis                 = None
        self.pixel_res              = None
        self.wavelength             = None
        
class zoneplate_object(wavefield_object):
    def __init__(self):
        self.focal_length           = None
        

def gen_zoneplate(dr=None,N=None,energy=None,material=None,thickness=None):
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
    
    # 120 um zoneplate in TPS 25A by XRnanotech
    # dr = 70e-9
    # N = 429
    # thickness = 1500E-9
    
    wavelength = tools.energy_eV_to_wavelength_m(energy)
    property = material_property.atomic_database('Au')
    n_refractivity = property.calculate_refractive_index(energy = energy,theta = 0)
    # diameter and focal length of the zone plate
    D = 4*N*dr
    f = (4*N*dr**2)/wavelength
    ## temporary 

    # E(z) = E0 * exp(i * n k z)
    # modulator = E(z)/E0 = exp(i * n k z)
    zp_modulation_factor = np.exp(1j * n_refractivity * 2*pi/wavelength * thickness)/np.exp(1j * 1 * 2*pi/wavelength * thickness)
    

    # create zp
    n = np.arange(N+1)
    rn = np.sqrt(n*wavelength*f+(n*wavelength)**2/4)
    pix_res = dr/2
    range = D*2
    pix_num = np.int32(np.round(range/pix_res))
    if np.mod(pix_num,2)==0:
        pix_num = pix_num + 1
    cen_idx = (pix_num-1)/2
    zp = np.ones([pix_num,pix_num],dtype = np.complex128)
    
    x_axis = (np.arange(pix_num)-cen_idx)*pix_res
    y_axis = x_axis*-1 # the direction of y is inverse of the row axis
    x_matrix,y_matrix = np.meshgrid(x_axis,y_axis)
    distance_map = np.sqrt(x_matrix**2 + y_matrix**2)
    
    for n_sn in np.flip(np.arange(0,N+1)):
        if np.mod(n_sn,2) == 0:
            zp[distance_map<rn[n_sn]] = zp_modulation_factor
        else:
            zp[distance_map<rn[n_sn]] = 1
            
    zp_object = zoneplate_object()
    zp_object.data = zp
    zp_object.x_axis = x_axis
    zp_object.z_axis = x_axis
    zp_object.pixel_res = pix_res
    zp_object.focal_length = f
    zp_object.wavelength = wavelength
    
    return zp_object


def wavefield_propagating(origin_wavefield_object = None, z = None):
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
    wavefield = origin_wavefield_object.data
    wavelength = origin_wavefield_object.wavelength
    pixel_res = origin_wavefield_object.pixel_res
    xi_axis = origin_wavefield_object.x_axis
    eta_axis = origin_wavefield_object.z_axis
    
    if wavefield.ndim == 2:
        wavefield = wavefield.reshape(1,wavefield.shape[0],wavefield.shape[1])
    
    k = 2*pi/wavelength
    
    xi_axis_res = pixel_res
    eta_axis_res = pixel_res
    
    xi,eta = np.meshgrid(xi_axis,eta_axis) #[m]
    U_measured_eta_size,U_measured_xi_size = wavefield[0].shape
    xp_res = np.abs(wavelength*z/U_measured_xi_size/xi_axis_res)
    yp_res = np.abs(wavelength*z/U_measured_eta_size/eta_axis_res)
    U_propagated_yp_size,U_propagated_xp_size = wavefield[0].shape
    xp_axis = (np.arange(U_propagated_xp_size)-U_propagated_xp_size/2)*xp_res
    yp_axis = (np.arange(U_propagated_yp_size)-U_propagated_yp_size/2)*yp_res
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
    
    if propagated_wavefield.shape[0] == 1:
        propagated_wavefield = propagated_wavefield[0]
        
    propagated_wavefield_object = wavefield_object()
    propagated_wavefield_object.data                   = propagated_wavefield
    propagated_wavefield_object.x_axis                 = propagated_x_axis
    propagated_wavefield_object.z_axis                 = propagated_y_axis
    propagated_wavefield_object.pixel_res              = xp_res
    propagated_wavefield_object.wavelength             = wavelength
    
    return propagated_wavefield_object