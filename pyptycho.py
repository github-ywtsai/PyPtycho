import numpy as np
import tools

class data_pretreatment_config:
    def __init__(self):
        self.clip_xcen              = None # start from 0
        self.clip_ycen              = None # start from 0
        self.clip_size              = None # should be a odd number
        self.saturation_threshold   = None
        
class pretreated_data:
    def __init__(self):
        self.data                   = None
        self.mask                   = None 
       
def pretreat_data(raw_data_object,data_pretreatment_config):
    # rearrange xcen
    if isinstance(data_pretreatment_config.clip_xcen, str):
        auto_detectred_cen = np.round(raw_data_object.header.BeamCenterX)
        data_pretreatment_config.clip_xcen = int(auto_detectred_cen)
        print('clip_xcen is set to %d automatically.'%(auto_detectred_cen))
    else:
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
    