import numpy as np

class TraceProperties:
    def __init__(self, trace, start, width, int_pts, per_amp=0.5, rli=1.0):
        """
        Parameters
        ----------
        trace : np.ndarray
            1-D numpy array of processed data (like `proData`).
        start : int
            Start index of the analysis window.
        width : int
            Width of the analysis window.
        int_pts : float
            Integration points (sampling interval).
        per_amp : float
            Proportion of amplitude for half-amp latency (default 0.5).
        rli : float
            RLI baseline value for this trace (default: 1.0).
        """
        self.trace = trace
        self.start = start
        self.width = width

        # truncate width if start + width exceeds data length
        if self.start + self.width > len(self.trace):
            self.width = len(self.trace) - self.start - 1
        self.int_pts = int_pts
        self.per_amp = per_amp
        self.rli = rli

        # Outputs
        self.max_amp = None
        self.max_amp_latency = None
        self.half_amp_latency = None
        self.max_amp_latency_pt = None
        self.half_amp_latency_decay = None
        self.half_width = None
        

        # Spike properties
        self.spike_start = None
        self.spike_end = None

        # Run calculations
        self.measure_properties()

    def measure_properties(self):
        """ Based on PhotoZ Data.cpp:measureProperties
            Calculates:
            1. Max Amp
            2. Max Amp Latency
            3. Half Amp Latency
            4. Half Width
              """
        num_pts = len(self.trace)

        #-------------------------------------------------------
        # 1. Max Amp
        # 2. Max Amp Latency
        self.max_amp = 0.0
        self.max_amp_latency = self.start + self.width

        for i in range(self.start, min(self.start + self.width + 1, num_pts)):
            if self.trace[i] > self.max_amp:
                self.max_amp = self.trace[i]
                self.max_amp_latency = i

        #-------------------------------------------------------
        # 3. Half Amp Latency
        if self.max_amp == 0:  # can't measure anything else if no response
            self.half_amp_latency = self.start
            return

        half_amp = self.max_amp * self.per_amp
        self.half_amp_latency = self.start

        for i in range(self.start, self.max_amp_latency + 1):
            if self.trace[i] == half_amp:
                self.half_amp_latency = float(i)
                break
            elif self.trace[i] > half_amp:
                if i == self.start:
                    self.half_amp_latency = float(i)
                    break
                denom = self.trace[i] - self.trace[i-1]
                if denom == 0:
                    self.half_amp_latency = float(i)
                else:
                    # interpolate
                    self.half_amp_latency = float(i) - (self.trace[i] - half_amp) / denom
                    #self.half_amp_latency = float(i) - 1 + (half_amp - self.trace[i-1]) / (self.trace[i] - self.trace[i-1])
                break
        
        # 4. Half Width = t(Decay to half amp) - t(Rise to half amp)
        # calculate time to reach halfAmp in the decay
        for i in range(self.max_amp_latency + 1, self.start + self.width + 1):
            if self.trace[i] < half_amp:
                if i == self.max_amp_latency + 1:
                    self.half_amp_latency_decay = float(i)
                    break
                denom = self.trace[i] - self.trace[i-1]
                if denom == 0:
                    self.half_amp_latency_decay = float(i)
                else:
                    # interpolate
                    self.half_amp_latency_decay = float(i) - (half_amp - self.trace[i]) / denom
                    #self.half_amp_latency_decay = float(i) - 1 + (self.trace[i-1] - half_amp) / (self.trace[i-1] - self.trace[i])
                break
        if self.half_amp_latency_decay is None:
            self.half_amp_latency_decay = float(self.start)
        self.half_width = self.half_amp_latency_decay - self.half_amp_latency
        if self.half_width is None or self.half_width < 0:
            self.half_width = 0
        
        # Convert points to ms
        self.max_amp_latency_pt = int(self.max_amp_latency)
        self.max_amp_latency *= self.int_pts
        self.half_amp_latency *= self.int_pts
        self.half_width *= self.int_pts


    def get_max_amp(self):
        return self.max_amp

    def get_max_amp_latency(self):
        return self.max_amp_latency

    def get_half_amp_latency(self):
        return self.half_amp_latency

    def get_spike_start(self):
        return self.spike_start

    def get_spike_end(self):
        return self.spike_end
    
    def get_half_amp_latency_decay(self):
        return self.half_amp_latency_decay
    
    def get_half_width(self):
        return self.half_width

    def get_SD(self):
        """ Get standard deviation. Mirrors PhotoZ implementation (Data.cpp:getSD) """
        '''double Data::getSD()
        {
            int i;
            int num=50;
            int startPt=10;
            double sum2=0,sum1=0;
            double data;

            for(i=startPt;i<startPt+num;i++)
            {
                data=proData[i];
                sum1+=data;
                sum2+=data*data;
            }

            return sqrt((sum2-sum1*sum1/num)/(num-1));
        }'''
        sum1 = 0.0
        sum2 = 0.0
        num = 50
        startPt = 10
        for i in range(startPt, startPt + num):
            data = self.trace[i]
            sum1 += data
            sum2 += data * data
        return np.sqrt((sum2 - sum1 * sum1 / num) / (num - 1))
    
    def get_SNR(self):
        return self.get_max_amp() / self.get_SD()
