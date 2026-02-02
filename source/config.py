import numpy as np

class Config(object): 
    def __init__(self, INTENSITIES, TYPE=0, NUM_HEART_BEATS=2):
        '''
        Define the Environment parameters of CT setup
        '''
     
        # Perform some sanity checks
        
        assert isinstance(INTENSITIES, np.ndarray), 'INTENSITIES must be a Nx1 numpy array'
        assert len(INTENSITIES.shape) == 2, 'INTENSITIES must be a Nx1 numpy array'
        assert isinstance(TYPE, int) and TYPE in [0,1,2], 'TYPE must be either 0, 1 or 2'
        assert isinstance(NUM_HEART_BEATS, float) and NUM_HEART_BEATS >= 0 and NUM_HEART_BEATS < 10, 'NUM_HEART_BEATS must be a float between 1 and 10'
#         assert isinstance(NUM_SDFS, int) and NUM_SDFS > 0 and NUM_SDFS < 5, 'NUM_SDFs should be positive integer not more than 5' 
        
        
        self.IMAGE_RESOLUTION = 128              # Resolution of the CT image
        self.GANTRY_VIEWS_PER_ROTATION = 720     # Number of views that the gantry clicks in a single 360 degree rotation
        self.HEART_BEAT_PERIOD = 1000            # Time (ms) it takes the heart to beat once
        self.GANTRY_ROTATION_PERIOD = 275        # Time (ms) it takes for the gantry to complete a single 360 degree rotation
        self.NUM_HEART_BEATS = NUM_HEART_BEATS   # Number of heart beats during the time HEART_BEAT_PERIOD
        self.INTENSITIES = INTENSITIES
        self.TYPE = TYPE
        '''
        NOTE: In the current setup, all of motion happens within the period HEART_BEAT_PERIOD. In case there are N hearbeats, then the time period of each heart beat is taken as HEART_BEAT_PERIOD/N. 
        '''
        
        '''
        Parameters for defining experimental setup
        '''
        if self.TYPE==0:
        # To run gantry for a single 360 degree rotation
            self.TOTAL_CLICKS = self.GANTRY_VIEWS_PER_ROTATION
            self.THETA_MAX = 360
            self.GANTRY2HEART_SCALE = (self.NUM_HEART_BEATS/self.THETA_MAX)*(self.GANTRY_ROTATION_PERIOD/self.HEART_BEAT_PERIOD)
        
        elif self.TYPE==1:
        # Otherwise, to run gantry for a single heart beat
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS)
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS)
            self.GANTRY2HEART_SCALE = 1/(self.THETA_MAX)
        
        elif self.TYPE==2:
        # Lastly, if you wish to run gantry to capture N heart beats then
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.GANTRY2HEART_SCALE = self.NUM_HEART_BEATS/(self.THETA_MAX)
        
        '''
        NeuralCT Hyper parameters
        '''
        self.SDF_SCALING = self.IMAGE_RESOLUTION/1.414  # Factor to scale NeuralCT's output to match G.T. SDF range of values
        self.BATCH_SIZE=25                       # Number of projections used in a single training iterations
        self.NUM_SDFS = self.INTENSITIES.shape[1]

class ConfigXCAT(object): 
    def __init__(self, INTENSITIES=None, NUM_SDFS=None,
                 IMAGE_RESOLUTION=None, GANTRY_VIEWS_PER_ROTATION=None,
                 HEART_BEAT_PERIOD=None, GANTRY_ROTATION_PERIOD=None):
        '''
        Define the Environment parameters of CT setup
        
        Args:
            INTENSITIES: Nx1 numpy array of organ intensities
            TYPE: Experiment type (0, 1, or 2)
            NUM_HEART_BEATS: Number of heart beats during scan
            IMAGE_RESOLUTION: Resolution of CT image (default: 128)
            GANTRY_VIEWS_PER_ROTATION: Number of projections per 360° rotation (default: 720)
            HEART_BEAT_PERIOD: Time (ms) for one heartbeat (default: 1000)
            GANTRY_ROTATION_PERIOD: Time (ms) for one gantry rotation (default: 275)
        '''
     
        # Perform some sanity checks
        if INTENSITIES is not None:
            assert len(INTENSITIES.shape) == 3, 'INTENSITIES must be a Nx1 numpy array'
        
        # Set defaults for optional parameters
        self.IMAGE_RESOLUTION = IMAGE_RESOLUTION if IMAGE_RESOLUTION is not None else 128
        self.GANTRY_VIEWS_PER_ROTATION = GANTRY_VIEWS_PER_ROTATION if GANTRY_VIEWS_PER_ROTATION is not None else 1000
        self.HEART_BEAT_PERIOD = HEART_BEAT_PERIOD if HEART_BEAT_PERIOD is not None else 1000
        self.GANTRY_ROTATION_PERIOD = GANTRY_ROTATION_PERIOD if GANTRY_ROTATION_PERIOD is not None else 250
        self.NUM_HEART_BEATS = 1
        self.INTENSITIES = INTENSITIES
        '''
        NOTE: In the current setup, all of motion happens within the period HEART_BEAT_PERIOD. In case there are N hearbeats, then the time period of each heart beat is taken as HEART_BEAT_PERIOD/N. 
        '''
        
        '''
        Parameters for defining experimental setup
        '''
        
        self.GANTRY2HEART_SCALE = (self.NUM_HEART_BEATS/360)*(self.GANTRY_ROTATION_PERIOD/self.HEART_BEAT_PERIOD)
        self.THETA_MAX = 1 / self.GANTRY2HEART_SCALE
        self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * self.THETA_MAX / 360)
        
        '''
        NeuralCT Hyper parameters
        '''
        self.SDF_SCALING = self.IMAGE_RESOLUTION/1.414  # Factor to scale NeuralCT's output to match G.T. SDF range of values
        self.BATCH_SIZE=25                       # Number of projections used in a single training iterations
        self.NUM_SDFS = NUM_SDFS if self.INTENSITIES is None else self.INTENSITIES.shape[1]
    
    @classmethod
    def from_xcat(cls, views_per_rotation, xcat_detectors, NUM_SDFS=None,
                  HEART_BEAT_PERIOD=None, GANTRY_ROTATION_PERIOD=None, INTENSITIES=None):
        '''
        Create Config from XCAT data parameters
        
        Args:
            INTENSITIES: Nx1 numpy array of organ intensities
            views_per_rotation: Number of projection angles in XCAT data
            xcat_detectors: Number of detectors in XCAT data
            HEART_BEAT_PERIOD: Time (ms) for one heartbeat (default: 1000)
            GANTRY_ROTATION_PERIOD: Time (ms) for one gantry rotation (default: 275)
        
        Returns:
            Config object initialized from XCAT parameters
        '''
        return cls(
            INTENSITIES=INTENSITIES,
            IMAGE_RESOLUTION=xcat_detectors,
            GANTRY_VIEWS_PER_ROTATION=views_per_rotation,
            HEART_BEAT_PERIOD=HEART_BEAT_PERIOD,
            GANTRY_ROTATION_PERIOD=GANTRY_ROTATION_PERIOD,
            NUM_SDFS=NUM_SDFS

        )
    
    def set_intensities(self, INTENSITIES):
        '''
        Set organ intensities and update related parameters
        
        Args:
            INTENSITIES: Nx1 numpy array of organ intensities
        '''
        assert len(INTENSITIES[0]) == 3, 'INTENSITIES must be a Nx1 numpy array'
        self.INTENSITIES = INTENSITIES[0]
        self.NUM_SDFS = len(self.INTENSITIES)