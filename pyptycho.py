import numpy as np
import tools

class data_pretreatment_config:
    def __init__(self):
        self.clip_xcen              = None # start from 0
        self.clip_ycen              = None # start from 0
        self.clip_size              = None # should be a odd number
        self.saturation_threshold   = None # for individual mask
        self.binning                = None # binning factor

class probe_gen_config:
    def __init__(self):
        self.probe_gen_mode         = None # prboe gen. mode: 'gaussian', 'focus', 'sim', 'adapt'
        
class pretreated_data_object:
    def __init__(self):
        self.data                   = None
        self.mask                   = None
        self.exp_pos_x              = None
        self.exp_pos_z              = None

class probe_object:
    def __init__(self):
        self.data                   = None
        self.x_axis                 = None
        self.z_axis                 = None
        self.pixel_res              = None
        
class object_object:
    def __init__(self):
        self.data                   = None
        self.x_axis                 = None
        self.z_axis                 = None
        self.pixel_res              = None
       
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
    
    ## binning section
    if data_pretreatment_config.binning != 1:
        print('Binning function is not ready yet.') 
        
    # tools.matrix_clip function will return a new matrix for the cliped matrix
    # pretreated_data = pretreated_data_object()
    data_temp = tools.matrix_clip(raw_data_object.data,data_pretreatment_config.clip_ycen,data_pretreatment_config.clip_xcen,data_pretreatment_config.clip_size)
    saturation_mask = data_temp >= data_pretreatment_config.saturation_threshold # individual mask
    pixel_mask = tools.matrix_clip(raw_data_object.header.PixelMask,data_pretreatment_config.clip_ycen,data_pretreatment_config.clip_xcen,data_pretreatment_config.clip_size) # mask configuration of detector
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
    
    return pretreated_data

    
    
    