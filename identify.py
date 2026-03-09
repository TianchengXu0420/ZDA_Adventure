import numpy as np
from itertools import combinations

class Identify:

    def __init__(self, Data=None):

        if Data is None:
            pass
        else:
            self.Data = Data

    def compute_snr(self, trace, startPt, numPt):
        '''
        Compute SNR value for a single trace.
        '''

        signal = np.max(trace[startPt:(startPt+numPt)])
        noise = np.std(trace[10:60], ddof=1)

        return signal / noise
    
    def single_neuron(self):
        '''
        Find all the combinations including the center pixel in a 3*3 area.
        '''

        coords = [(dr, dc) for dr in [-1,0,1] for dc in [-1,0,1]]
        center = (0, 0)
        roi_shapes = []

        for k in range(1, 10):
            for comb in combinations(coords, k):

                if center not in comb:
                    continue
                else:
                    roi_shapes.append(comb)

        self.roi_shapes = roi_shapes

        return roi_shapes
    
    def optimize_map(self, Data_ave, startPt, numPt):
        '''
        For each pixel in the 80*80 map, find its best roi shape with the highest SNR value.
        '''

        roi_shapes = self.single_neuron()
        H, W, T = Data_ave.shape

        best_snrs = np.zeros((H,W))
        best_rois = [[None]*W for _ in range(H)]

        for r in range(1,H-1):
            for c in range(1,W-1):

                best_snr = -np.inf
                best_shape = None

                for shape in roi_shapes:

                    pixels = [(r+dr, c+dc) for dr,dc in shape]

                    selected = np.array([Data_ave[x,y] for x,y in pixels])
    
                    avg_trace = np.mean(selected, axis=0)

                    snr = self.compute_snr(trace=avg_trace, startPt=startPt, numPt=numPt)

                    if snr > best_snr:

                        best_snr = snr
                        best_shape = pixels

                best_rois[r][c] = best_shape
                best_snrs[r,c] = best_snr

        return best_rois, best_snrs
    
    def candidates(self, Data_ave, cutoff, startPt, numPt):
        '''
        Find the 'good' 3*3 areas from the whole 80*80 map.
        '''

        rois, snrs = self.optimize_map(Data_ave=Data_ave, startPt=startPt, numPt=numPt)
        candidates = []

        for i in range(80):
            for j in range(80):

                candidates.append([int(i), int(j), snrs[i][j], rois[i][j]])

        third_values = [x[2] for x in candidates]
        percent = np.percentile(third_values, cutoff)

        candidates = [x for x in candidates if x[2] >= percent]

        return candidates
    
    def select_non_overlap(self, candidates, snr_map):
        '''
        Avoid overlapping 3*3 areas.
        '''

        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

        selected = []
        occupied = np.zeros_like(snr_map, dtype=bool)

        for r, c, snr_value, comb in candidates:
            r = int(r)
            c = int(c)
            if not np.any(occupied[r-1:r+2, c-1:c+2]):
                selected.append([r, c])
                occupied[r-1:r+2, c-1:c+2] = True

        return selected