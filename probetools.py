import numpy as np
from numpy import pi


class exit_wave_object():
    def __init__(self):
        self.data                   = None
        self.x_axis                 = None
        self.z_axis                 = None
        self.pixel_res              = None

def gen_zoneplate(dr=50e-9,N=300,wavelength=0.8265e-10,material='Au',thickness=600e-9):
    # dr: the min. pitch
    # N: the period for the zone plate design
    # using formula from Attwood Soft X-rays and Extereme Ultraviolet Radiation
    # D_cs: diameter of the central stop [m]
    # T_cs: transmittance of the central stop
    # off_focal: the sample position referring to the focal plane [m]
    # material: zoneplate made of
    # thickness: thickness of the zoneplate [m]
    
    # 96 um zoneplate in TPS 25A
    # dr = 30e-9
    # N = 800
    
    # 60 um zoneplate in TPS 25A
    # dr = 50e-9
    # N = 300 
    
    # diameter and focal length of the zone plate
    D = 4*N*dr
    f = (4*N*dr**2)/wavelength
    
    ## temporary 
    # calculate refractivity
    # n(wavelength) = 1 - delta(wavelength) + i beta(wavelength)
    f1 = 74.9382 # [e/atom] from NIST
    f2 = 6.0726 # [e/atom] % from NIST
    re = 2.8179403e-15 # classical electron rauius [m]
    Na = 6.02214129e23 # Avogadro constant [1/mol]
    Ma = 196.966569 # molar mass [g/mol]
    rho = 19.32 # density [g/cm^3]
    rho = rho / (1e-2)**3 # convert unit from [g cm-3] to [g m-3]
    na = rho*Na/Ma # number density
    delta = na*re*wavelength**2/2/pi*f1
    beta =  na*re*wavelength**2/2/pi*f2
    n_refractivity = 1 - delta + 1j * beta
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
#        else:
#            zp[distance_map<rn[n_sn]] = 0
            
    zp_object = exit_wave_object()
    zp_object.data = zp
    zp_object.x_axis = x_axis
    zp_object.z_axis = x_axis
    zp_object.pixel_res = pix_res
    
    return zp_object