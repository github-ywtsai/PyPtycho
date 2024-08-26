import numpy as np
import tools
import probetools

# for diffraction pattern
class data_pretreatment_config:
    def __init__(self):
        self.clip_xcen              = None # start from 0
        self.clip_ycen              = None # start from 0
        self.clip_size              = None # should be a odd number
        self.saturation_threshold   = None # for individual mask
        self.binning                = None # binning factor
        
class pretreated_data_object:
    def __init__(self):
        self.data                   = None
        self.mask                   = None
        self.exp_pos_x              = None
        self.exp_pos_z              = None
        self.exp_pos_x_shift        = None
        self.exp_pos_z_shift        = None
        self.pixel_size             = None
        self.wavelength             = None
        self.detector_distance      = None
        self.energy                 = None

# for probe
class probe_gen_config:
    def __init__(self):
        self.mixture_state          = None # number of probe
        self.gen_method             = None # zoneplate, adapt else.

        self.zoneplate_config = zoneplate_config()
        
class zoneplate_config:
    def __init__(self):
        self.dr                     = None
        self.N                      = None
        self.energy                 = None
        self.material               = None
        self.thickness              = None
        self.defocal                = None

class probe_object(probetools.wavefield_object):
    def __init__(self):
        super().__init__()
        self.attribute              = 'probe'
        self.cdi_window             = None

# for object     
class obj_gen_config:
    def __init__(self):
        self.extend_ratio           = 0.1 # area extend, default 10% 
        self.transmission_range     = [0.95, 1] # random range of the transmission

class obj_object(probetools.wavefield_object):
    def __init__(self):
        super().__init__()
        self.attribute = 'object'
        
def gen_probe(pretreated_data_object,probe_gen_config):
    probe = probe_object()
    data = np.zeros([probe_gen_config.mixture_state,pretreated_data_object.clip_size,pretreated_data_object.clip_size]) *  0j
    pixel_res = tools.cal_real_space_pixel_res(wavelength = pretreated_data_object.wavelength , detector_distance = pretreated_data_object.detector_distance , clip_size = pretreated_data_object.clip_size , pixel_size = pretreated_data_object.pixel_size)
    cdi_window = pixel_res * pretreated_data_object.clip_size
    
    cen_idx = np.int32((pretreated_data_object.clip_size - 1)/2)
    x_axis = (np.arange(pretreated_data_object.clip_size)-cen_idx) * pixel_res 
    z_axis = np.flip(x_axis)
    
    probe.data = data
    probe.pixel_res = pixel_res
    probe.x_axis = x_axis
    probe.z_axis = z_axis
    probe.cdi_window = cdi_window
    probe.energy = pretreated_data_object.energy
    probe.wavelength = pretreated_data_object.wavelength

    print('Probe shape: {}'.format(probe.data.shape))
    print('CDI window: {}'.format(tools.show_length_with_unit(cdi_window)))
    print('Probe pixel resolution: {}'.format(tools.show_length_with_unit(pixel_res)))
    
    ## start adapt wavefield
    if probe_gen_config.gen_method == 'zoneplate':
        dr          = probe_gen_config.zoneplate_config.dr
        N           = probe_gen_config.zoneplate_config.N
        energy      = probe_gen_config.zoneplate_config.energy
        material    = probe_gen_config.zoneplate_config.material
        thickness   = probe_gen_config.zoneplate_config.thickness
        defocal     = probe_gen_config.zoneplate_config.defocal
        zp = probetools.gen_zoneplate(dr=dr,N=N,energy=energy,material=material,thickness=thickness)
        zpp = probetools.wavefield_propagating(origin_wavefield_object = zp, z = zp.focal_length+defocal,window = None)
        
        matched_zpp = probetools.wavefield_matching(reference_wavfield_object = probe, target_wavefield_object = zpp)       
        probe.data[0] = matched_zpp.data[0]
        
        # arrange 2nd, 3rd,...etc probes
        if probe_gen_config.mixture_state > 1:
            main_probe_amp = np.abs(probe.data[0])
            main_probe_phase = np.angle(probe.data[0])
            main_probe_intensity = np.power(main_probe_amp,2)
            main_intensity_sum = np.sum(main_probe_intensity)
            random_intensity = np.random.rand(probe_gen_config.mixture_state-1,pretreated_data_object.clip_size,pretreated_data_object.clip_size)
            random_intensity_sum = 0.5 * pretreated_data_object.clip_size * pretreated_data_object.clip_size # the mean of a random matrix is approximate to 0.5
            rand_probe_intensity = (random_intensity*main_intensity_sum/random_intensity_sum + main_probe_intensity)/2 *0.01 # generate a 1% probe (intensity)
            rand_probe_amp = np.sqrt(rand_probe_intensity)
            rand_probe = rand_probe_amp * np.exp(1j*main_probe_phase)
            probe.data[1:] = rand_probe
    
    return probe
    
def gen_obj(pretreated_data_object,obj_gen_config):
    obj = obj_object()
    pixel_res = tools.cal_real_space_pixel_res(wavelength = pretreated_data_object.wavelength , detector_distance = pretreated_data_object.detector_distance , clip_size = pretreated_data_object.clip_size , pixel_size = pretreated_data_object.pixel_size)
    cdi_window = pixel_res * pretreated_data_object.clip_size
    
    obj_x_range = ( pretreated_data_object.exp_pos_x.max()-pretreated_data_object.exp_pos_x.min() + cdi_window ) * (1+obj_gen_config.extend_ratio)
    obj_z_range = ( pretreated_data_object.exp_pos_z.max()-pretreated_data_object.exp_pos_z.min() + cdi_window ) * (1+obj_gen_config.extend_ratio)

    
    obj_col_size = np.round(obj_x_range/pixel_res).astype(int)
    obj_row_size = np.round(obj_z_range/pixel_res).astype(int)

    if tools.iseven(obj_row_size):
        obj_row_size = obj_row_size + 1
    if tools.iseven(obj_col_size):
        obj_col_size = obj_col_size + 1
    
    obj_row_cen_idx = ((obj_row_size - 1)/2).astype(int)
    obj_col_cen_idx = ((obj_col_size - 1)/2).astype(int)
    
    x_axis = (np.arange(obj_col_size) - obj_col_cen_idx)*pixel_res
    z_axis = np.flip((np.arange(obj_row_size) - obj_row_cen_idx)*pixel_res)
    
    data = (np.random.uniform(low = np.min(obj_gen_config.transmission_range),high = np.max(obj_gen_config.transmission_range),size = [obj_row_size,obj_col_size]))*0j
    
    obj.data = data
    obj.x_axis = x_axis
    obj.z_axis = z_axis
    obj.pixel_res = pixel_res
    obj.energy = pretreated_data_object.energy
    obj.wavelength = pretreated_data_object.wavelength
    
    print('Object shape: {}'.format(obj.data.shape))
    print('Object FOV: {}'.format(tools.show_length_with_unit(obj_col_size*pixel_res)),'X {}'.format(tools.show_length_with_unit(obj_row_size*pixel_res)))
    print('Object pixel resolution: {}'.format(tools.show_length_with_unit(pixel_res)))
    
    return obj
       
def pretreat_data(raw_data_object,data_pretreatment_config):
    # rearrange xcen
    if isinstance(data_pretreatment_config.clip_xcen, str):
        auto_detectred_cen = np.round(raw_data_object.header.BeamCenterX)
        data_pretreatment_config.clip_xcen = int(auto_detectred_cen)
        print('clip_xcen is set to %d automatically.'%(auto_detectred_cen))
    else:
        print(data_pretreatment_config)
        data_pretreatment_config.clip_xcen = int(np.round(data_pretreatment_config.clip_xcen))
        print('clip_xcen is set to %d manually.'%(data_pretreatment_config.clip_xcen))
    
    # rearrange ycen
    if isinstance(data_pretreatment_config.clip_ycen, str):
        auto_detectred_cen = np.round(raw_data_object.header.BeamCenterY)
        data_pretreatment_config.clip_ycen = int(auto_detectred_cen)
        print('clip_ycen is set to %d automatically.'%(auto_detectred_cen))
    else:
        data_pretreatment_config.clip_ycen = int(np.round(data_pretreatment_config.clip_ycen))
        print('clip_ycen is set to %d manually.'%(data_pretreatment_config.clip_ycen))
        
    # rearrange clip_size
    data_pretreatment_config.clip_size = int(np.round(data_pretreatment_config.clip_size))
    if tools.iseven(data_pretreatment_config.clip_size):
        data_pretreatment_config.clip_size = data_pretreatment_config.clip_size + 1
        print('clip_size must be a odd number. which is modified to %d.'%(data_pretreatment_config.clip_size))
        
    # rearrange saturation_threshold
    if isinstance(data_pretreatment_config.saturation_threshold, str):
        data_pretreatment_config.saturation_threshold = raw_data_object.header.SaturationIntensity
        print('saturation_threshold is set to %d automatically.'%(data_pretreatment_config.saturation_threshold))
    else:
        data_pretreatment_config.saturation_threshold = data_pretreatment_config.saturation_threshold
        print('saturation_threshold is set to %d manually.'%(data_pretreatment_config.saturation_threshold))
    
    
    wavelength = raw_data_object.header.Wavelength
    energy = raw_data_object.header.Energy
    pixel_size = raw_data_object.header.XPixelSize
    detector_distance = raw_data_object.header.DetectorDistance
    clip_size = data_pretreatment_config.clip_size
    clip_xcen = data_pretreatment_config.clip_xcen
    clip_ycen = data_pretreatment_config.clip_ycen
    raw_data = np.copy(raw_data_object.data)
    raw_pixel_mask = np.copy(raw_data_object.header.PixelMask)
    
        ## binning section
    if data_pretreatment_config.binning != 1:
        # when benning on, change parameters here
        print('Binning function is not ready yet.')
        
        
    # tools.matrix_clip function will return a new matrix for the cliped matrix
    # pretreated_data = pretreated_data_object()
    data_temp = tools.matrix_clip(raw_data,clip_ycen,clip_xcen,clip_size)
    saturation_mask = data_temp >= data_pretreatment_config.saturation_threshold # individual mask
    pixel_mask = tools.matrix_clip(raw_pixel_mask,clip_ycen,clip_xcen,clip_size) # mask configuration of detector
    mask_temp = np.logical_or(saturation_mask,pixel_mask)
    
    # rearrange exposure position
    # Definition:
    # 1. Use the central-of-mass of the the scanning points as the center.
    # 2. Transform all scanning positions from positions to shifts.
    # 3. Redirect all shifts to "exposure positions" on the sample, where the coordinate on the sample is a XY Cartesian Coordinate System.    
    readback_x = raw_data_object.scandata.readback_x
    readback_z = raw_data_object.scandata.readback_z
    readback_x_cen = np.mean(readback_x)
    readback_z_cen = np.mean(readback_z)
    x_shift = readback_x - readback_x_cen
    z_shift = readback_z - readback_z_cen
    exp_pos_x = x_shift* -1 * raw_data_object.scandata.motor_direction_x
    exp_pos_z = z_shift* -1 * raw_data_object.scandata.motor_direction_z
    
    pretreated_data = pretreated_data_object()
    pretreated_data.data = np.ma.masked_array(data_temp, mask = mask_temp)
    pretreated_data.mask = mask_temp
    pretreated_data.exp_pos_x = exp_pos_x
    pretreated_data.exp_pos_z = exp_pos_z
    pretreated_data.exp_pos_x_shift = np.zeros(exp_pos_x.shape)
    pretreated_data.exp_pos_z_shift = np.zeros(exp_pos_z.shape)

    pretreated_data.pixel_size              = pixel_size
    pretreated_data.wavelength              = wavelength
    pretreated_data.energy                  = energy
    pretreated_data.detector_distance       = detector_distance
    pretreated_data.clip_size               = clip_size
    
    return pretreated_data

    
    
    