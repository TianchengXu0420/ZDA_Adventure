import numpy as np
import struct


class ROIFileReader:
    """ Reads ROI data from PhotoZ dat file, as diode numbers """
    def __init__(self, filename, convert_diode_nums=True, w=80):
        self.w = w  # width of image for diode num to (x,y) conversion
        self.rois = []
        if filename is not None:
            self.read_file(filename)
        if convert_diode_nums:
            self.rois = [self.diode_num_to_points(r) for r in self.rois]

    """ Format:
    < # ROIs >
    < Region #1 index (0) >
    < Region #1 size >
    < Region #1 index (0) >   needed for unknown reason
    < Region #1 list start >
    ...
    < Region #1 list end >
    < Region #2 index (1) >
    < Region #2 size >
    < Region #2 index (1) >   needed for unknown reason
    < Region #2 list start >
    ...
    < Region #2 list end >
    ...
    """
    def get_roi_list(self):
        return self.rois

    def read_file(self, filename):
        """ populates rois as a doubly-nested
            list of diode numbers from file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            num_regions = int(lines[0])
            # print("File has", num_regions, "regions.")
            i = 1
            for j in range(num_regions):
                region_size = int(lines[i+1]) - 1
                # print("Region has", region_size, "pixels")
                i += 3  # skip index and other extra line
                next_region = []
                for k in range(region_size):
                    next_region.append(int(lines[i]))
                    # print(lines[i], "(i = " + str(i) + ")")
                    i += 1
                self.rois.append(next_region)
            if i < len(lines) - 1:
                print("Unread lines:", lines[i:])

    def diode_num_to_points(self, diode_numbers):
        pts = []
        for dn in diode_numbers:
            # x and y appear flipped to match image orientation
            x_px = dn % self.w  # looks like column
            y_px = int(dn / self.w)  # looks like row
            pts.append([x_px, y_px]) # x is off by 1 for some reason!
        return pts
    
    def merge_overlapping_rois(self):
        '''
        Merge overlapping ROIs into single ROIs.
        '''
        merged_rois = []
        for roi in self.rois:
            if self.convert_diode_nums:
                roi_set = set(tuple(pt) for pt in roi)
            else:
                roi_set = set(roi)
            merged = False
            for m_roi in merged_rois:
                if self.convert_diode_nums:
                    m_roi_set = set(tuple(pt) for pt in m_roi)
                else:
                    m_roi_set = set(m_roi)
                if not roi_set.isdisjoint(m_roi_set):
                    # Overlap found, merge
                    merged_roi_set = roi_set.union(m_roi_set)
                    m_roi.clear()
                    m_roi.extend([list(pt) for pt in merged_roi_set])
                    merged = True
                    break
            if not merged:
                merged_rois.append(roi)
        self.rois = merged_rois


class TraceSelector:
    
    """ Selects traces from Data based on ROI list """
    def __init__(self, data):
        self.data = data  # Data should be Trials x Height x Width x Points

    def get_trace_from_pixel(self, x, y, trial=None):
        '''
        Get traces from specified pixel.
        '''
        Data = self.get_data()
        
        if trial is None:
            # average over trial axis
            return np.mean(Data[:, y, x, :], axis=0)
        return Data[trial, y, x, :]

    def get_trace_from_roi(self, roi, trial=None):
        '''' An roi is a list of (x,y) tuples or [x,y] lists.
         Get traces from specified roi (mean over pixels in roi).
        '''
        roi = np.array(roi)
        Data = self.data
        if trial is None:
            # average over trial axis
            return np.mean(Data[:, roi[:, 1], roi[:, 0], :], axis=(0, 1))
        return Data[trial, roi[:, 1], roi[:, 0], :]


class DataLoader:
    
    def __init__(self, filedir, number_of_points_discarded=24):
        '''
        Initialize the Data and Parameters.
        Preparation for further processing.
        '''
        self.number_of_points_discarded = number_of_points_discarded
        self.scale_amplitude = 3.2768
    
        # Important Index of the ZDA Data.
        self.data, metadata, self.rli, self.supplyment = self.from_zda_to_numpy(filedir)
        self.filedir = filedir
        self.meta = metadata
        
        # The four dimensions of the Data Array.
        self.trials = metadata['number_of_trials']
        self.points = metadata['points_per_trace'] - self.number_of_points_discarded
        self.width = metadata['raw_width']
        self.height = metadata['raw_height']
           
    def from_zda_to_numpy(self, filedir):
        '''
        Read ZDA file and convert the data into numpy array which can be used in Python.
        Including RLI Data, MetaData, Raw Data, and Supplymental Data.
        '''
        
        # Load and read the binary ZDA file.
        file = open(filedir, 'rb')
        
        # Read in different scales.
        chSize = 1
        shSize = 2
        nSize = 4
        tSize = 8
        fSize = 4
        
        # MetaData.
        metadata = {}
        metadata['version'] = (file.read(chSize))
        metadata['slice_number'] = (file.read(shSize))
        metadata['location_number'] = (file.read(shSize))
        metadata['record_number'] = (file.read(shSize))
        metadata['camera_program'] = (file.read(nSize))

        metadata['number_of_trials'] = (file.read(chSize))
        metadata['interval_between_trials'] = (file.read(chSize))
        metadata['acquisition_gain'] = (file.read(shSize))
        metadata['points_per_trace'] = (file.read(nSize))
        metadata['time_RecControl'] = (file.read(tSize))

        metadata['reset_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['reset_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['shutter_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['stimulation1_onset'] = struct.unpack('f', (file.read(fSize)))[0]
        metadata['stimulation1_duration'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['stimulation2_duration'] = struct.unpack('f',(file.read(fSize)))[0]

        metadata['acquisition_onset'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['interval_between_samples'] = struct.unpack('f',(file.read(fSize)))[0]
        metadata['raw_width'] = (file.read(nSize))
        metadata['raw_height'] = (file.read(nSize))

        for k in metadata:
            if k in ['interval_between_samples',] or 'onset' in k or 'duration' in k:
                pass # any additional float processing can go here
            elif k == 'time_RecControl':
                pass # TO DO: timestamp processing
            else:
                metadata[k] = int.from_bytes(metadata[k], "little") # endianness

        num_diodes = metadata['raw_width'] * metadata['raw_height']
        
        file.seek(1024, 0)
        
        # RLI 
        rli = {}
        rli['rli_low'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_high'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        rli['rli_max'] = [int.from_bytes(file.read(shSize), "little") for _ in range(num_diodes)]
        
        # each FP has an RLI? discard 3 * 8 shorts from top
        for _ in range(3 * 8):
            _ = file.read(shSize)

        # Raw Data.
        raw_data = np.zeros((metadata['number_of_trials'],
                             metadata['points_per_trace'],
                             metadata['raw_width'],
                             metadata['raw_height'])).astype(int)
        # Supplymental Data (analog input (aka FP) data)
        supplyment = np.zeros(((metadata['number_of_trials']-1) *8, metadata['points_per_trace']))
        
        for i in range(metadata['number_of_trials']):
            '''for i_fp in range(8):
                for k in range(metadata['points_per_trace']):
                    pt = file.read(shSize)
                    if not pt:
                        print("Ran out of points.",len(raw_data))
                        file.close()
                        return raw_data, metadata, rli, None
                    supplyment[i, i_fp, k] = int.from_bytes(pt, "little")'''
            for jw in range(metadata['raw_width']):
                for jh in range(metadata['raw_height']):
                    for k in range(metadata['points_per_trace']):
                        pt = file.read(shSize)
                        if not pt:
                            print("Ran out of points.",len(raw_data))
                            file.close()
                            return raw_data, metadata, rli, None
                        raw_data[i,k,jw,jh] = int.from_bytes(pt, "little")
        

        for i in range(((metadata['number_of_trials']-1) * 8)):
            for j in range(metadata['points_per_trace']):
                pt = file.read(shSize)
                supplyment[i][j] = int.from_bytes(pt, "little")

        file.close()
        
        return raw_data, metadata, rli, supplyment
    
    def discard_and_rearrange(self):
        '''
        Discard and rearrange the useless (or noised) pixels and points.
        '''
        
        # Number of the Noised Data Points is self.number_of_points_discarded
        
        # Discard the Noised Data Points.
        Data = np.copy(self.data[:, self.number_of_points_discarded:, :, :])
        Supplyment = np.copy(self.supplyment[:, self.number_of_points_discarded:])
        Data_rearrange = np.zeros((self.trials, self.height, self.width, self.points))
        
        # Rearrange the Data.
        for i in range(self.trials):
            for j in range(self.height):
                for k in range(self.width):
                    
                    Data_rearrange[i][j][k] = Data[i, :, j, k]
        
        return Data_rearrange, Supplyment
    
    def fix_and_supply(self):
        '''
        Fix and supply the Data.
        '''
        
        # Load the Rearranged Data.
        Data_Raw, Supplyment = self.discard_and_rearrange()
        
        # Fix the Data.
        for i in range(self.trials):
            if i == 0:
                pass
            elif i!=0 and i<(self.trials-1):
                Data = np.copy(Data_Raw[i])
                Data_rearrange = Data.reshape(self.height*self.width, self.points)

                if i==1:
                    Data_fp = Data_rearrange[:8, self.points]
                
                Data_fix = np.delete(Data_rearrange, np.arange(0, (i*8), step=1), axis=0)
                
                Data_supply = np.copy(Data_Raw[i+1])
                Data_supply = Data_supply.reshape(self.height*self.width, self.points)
                Data_supply = Data_supply[:(i*8), :]
                
                Data_fix = np.concatenate((Data_fix, Data_supply), axis=0)
                Data_fix = Data_fix.reshape(self.height, self.width, self.points)
                
                Data_Raw[i] = Data_fix
            elif i == self.trials-1:
                Data = np.copy(Data_Raw[i])
                Data_rearrange = Data.reshape(self.height*self.width, self.points)
                Data_fix = np.delete(Data_rearrange, np.arange(0, (i*8), step=1), axis=0)
                
                Data_fix = np.concatenate((Data_fix, Supplyment), axis=0)
                Data_fix = Data_fix.reshape(self.height, self.width, self.points)
                
                Data_Raw[i] = Data_fix

        # invert and scale amplitude
        Data_Raw = -Data_Raw / self.scale_amplitude
        
        return Data_Raw, Data_fp
    
    def clamp(self):
        '''
        Clamp the first point of each pixel at zero. 
        '''
        
        # Load the Data.
        Data, _ = self.fix_and_supply()
        
        for i in range(self.trials):
            for j in range(self.height):
                for k in range(self.width):
                    
                    Data[i][j][k] = Data[i, j, k, :] - Data[i, j, k, 0]
                    
        return Data
    
    def get_data(self, rli_division=True):
        # TO DO: implement rli_division functionality
        if rli_division:
            print("rli_division functionality not yet implemented: utility.py:get_data method")
        return self.clamp()

    def get_index(self):    
        return self.meta
    
    def get_rli(self):
        return self.rli

    def get_fp(self):

        _, fp = self.fix_and_supply()
        return fp
