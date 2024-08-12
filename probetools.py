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
    zp = np.zeros([pix_num,pix_num],dtype = np.complex128)
    
    x_axis = (np.arange(pix_num)-cen_idx)*pix_res
    y_axis = x_axis*-1 # the direction of y is inverse of the row axis
    x_matrix,y_matrix = np.meshgrid(x_axis,y_axis)
    distance_map = np.sqrt(x_matrix**2 + y_matrix**2)
    
    for n_sn in np.flip(np.arange(0,N+1)):
        if np.mod(n_sn,2) == 0:
            zp[distance_map<rn[n_sn]] = zp_modulation_factor
        else:
            zp[distance_map<rn[n_sn]] = 0
            
    zp_object = zoneplate_object()
    zp_object.data = zp
    zp_object.x_axis = x_axis
    zp_object.z_axis = x_axis
    zp_object.pixel_res = pix_res
    zp_object.focal_length = f
    zp_object.wavelength = wavelength
    
    return zp_object