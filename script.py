import eiger

# load raw data from a bluesky scanning file
scan_file_path = 'c:\\SYNC_main\\NSRRC_TPS 25A\\programs\\test_data\\ptychography_seimanstart 7nm\\sample02251103-scan_id-485-primary.csv'
raw_data = eiger.bluesky_scanning_data().load_data(scan_file_path)




