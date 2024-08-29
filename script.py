import eiger
import pyptycho as pp
import tools

## load raw data from a bluesky scanning file
scan_file_path = 'c:\\SYNC_main\\NSRRC_TPS 25A\\programs\\test_data\\ptychography_seimanstart 7nm\\sample02251103-scan_id-485-primary.csv'
raw_data = eiger.load_bluesky_exp_data(scan_file_path)

## set data pretreatment config
data_pretreatment_config = pp.data_pretreatment_config()
data_pretreatment_config.clip_xcen              = 'auto' # 'auto' for adapting xcen from h5 header or a integer for setting manually 
data_pretreatment_config.clip_ycen              = 'auto' # 'auto' for adapting ycen from h5 header or a integer for setting manually 
data_pretreatment_config.clip_size              = 401 # must be a odd number
data_pretreatment_config.saturation_threshold   = 'auto' # 'auto' for adapting SaturationIntensity from h5 header or a integer for setting manually 
data_pretreatment_config.binning                = 2 # binning factor
## pretreat data
pretreated_data = pp.pretreat_data(raw_data,data_pretreatment_config)

## create probe
probe_gen_config = pp.probe_gen_config()
probe_gen_config.normalization                  = True
probe_gen_config.mixture_state                  = 3
probe_gen_config.gen_method                     = 'zoneplate' #'zoneplate'

# for zoneplate config
probe_gen_config.zoneplate_config.dr            = 70e-9 # in meter
probe_gen_config.zoneplate_config.N             = 286
probe_gen_config.zoneplate_config.energy        = pretreated_data.energy
probe_gen_config.zoneplate_config.thickness     = 1500e-6 # in meter
probe_gen_config.zoneplate_config.material      = 'Au'
probe_gen_config.zoneplate_config.defocal       = 200e-6 # in meter

probe = pp.gen_probe(pretreated_data,probe_gen_config)

## create object
obj_gen_config = pp.obj_gen_config()
obj = pp.gen_obj(pretreated_data,obj_gen_config)




