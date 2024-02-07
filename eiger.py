import h5py
import hdf5plugin
import os
import numpy as np

def read_header(master_fp):
    if not os.path.isfile(master_fp):
        print('File does not exist.')
        return

    fid = h5py.File(master_fp,'r') # open file object
    
    header = dict()
    # field can be browsed by list(fid.keys()), list(fid['entry/data/'].keys()) etc.

    # find max frame number
    dset = list(fid['entry/data/'].items())
    for dsn in range(len(dset)):
        if dset[dsn][1] == None:
            TotalFrame = dset[dsn-1][1].attrs['image_nr_high']
            break
    header['h5MasterFilePath']     = master_fp
    header['BitDepthImage']        = fid['/entry/instrument/detector/bit_depth_image'][()]
    header['SaturationIntensity']  = fid['/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff'][()]
    header['XPixelsInDetector']    = fid['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'][()]
    header['YPixelsInDetector']    = fid['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'][()]
    header['CountTime']            = fid['/entry/instrument/detector/count_time'][()]
    header['DetectorDistance']     = fid['/entry/instrument/detector/detector_distance'][()]
    header['XPixelSize']           = fid['/entry/instrument/detector/x_pixel_size'][()]
    header['YPixelSize']           = fid['/entry/instrument/detector/y_pixel_size'][()]
    header['Wavelength']           = fid['/entry/instrument/beam/incident_wavelength'][()]*1E-10 # convert from A to meter
    header['BeamCenterX']          = fid['/entry/instrument/detector/beam_center_x'][()]
    header['BeamCenterY']          = fid['/entry/instrument/detector/beam_center_y'][()]
    header['PixelROI']             = np.invert(fid['/entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool)) # convert the mask to logical array
    header['ManualROI']            = np.ones([header['YPixelsInDetector'],header['XPixelsInDetector']],dtype = bool)
    header['ROI']                  = None
    header['TotalFrame']           = TotalFrame

    
    fid.close()
    return header

def read_frame(master_fp,req):
    # req is the frame number
    # list(fid['entry/data/data_000002'].attrs) can show the attrs field name
    # fid['entry/data/data_000002'].attrs['image_nr_high'] can show the value
    
    fid = h5py.File(master_fp,'r') # open file object
     
    if req <= 0 :
        print('Request frame index <= 0.')
        return
     
    dset = list(fid['entry/data/'].items())
    fid.close()
    
    for dsn in range(len(dset)):
        if dset[dsn][1] == None:
            print('Request frame index > maximum frame number.')
            break    
        
        image_nr_low = dset[dsn][1].attrs['image_nr_low']
        image_nr_high = dset[dsn][1].attrs['image_nr_high']
        sn_list = np.arange(image_nr_low,image_nr_high+1)
        target_ind = np.where(sn_list == req)[0]
        if target_ind.size == 0:
            continue
        else:
            target_ind = target_ind[0]
            data = dset[dsn][1][target_ind]
            return data