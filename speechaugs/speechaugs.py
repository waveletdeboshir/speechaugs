import torch
import librosa
import colorednoise
import numpy as np

from albumentations.core.transforms_interface import BasicTransform

class BaseWaveformTransform(BasicTransform):

    @property
    def targets(self):
        return {"waveform": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params



class TimeStretchLibrosa(BaseWaveformTransform):
    """ 
        Speed up or slow down a single-channel waveform.
        
        Parameters:
        p: Default=0.5
        max_duration: maximum audio length (in seconds), Default=10
        sr: sample rate, Default=16000
    """

    def __init__(self, always_apply=False, p=0.5, sr=16000, max_duration=10):

        super(TimeStretchLibrosa, self).__init__(always_apply, p)

        self.max_duration = max_duration
        self.sr = sr

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        
        rate = np.random.uniform(0.5, 2.0) # rate < 1.0 -- slow down, rate > 1.0 -- speed up

        if waveform.shape[1]/rate>self.max_duration*self.sr:
            rate = np.random.uniform(1.0, 2.0) # If length is greater than maximal then we speed up

        waveform = librosa.effects.time_stretch(waveform[0].numpy(), rate)
                
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)

class PitchShiftLibrosa(BaseWaveformTransform):
    """ 
        Shift a pitch up or down by n semitones. 
        
        
        Parameters:
        p: Default=0.5
        sr: sample rate, Default=16000    
    """
    def __init__(self, always_apply=False, p=0.5, sr=16000):

        super(PitchShiftLibrosa, self).__init__(always_apply, p)

        self.sr = sr


    def apply(self, waveform, **params):
        waveform = waveform.clone()

        n_steps = np.random.randint(-10, 10) # n_steps < 0 -- shift down, n_steps > 0 -- shift up
        waveform = librosa.effects.pitch_shift(waveform[0].numpy(), self.sr, n_steps=n_steps, bins_per_octave=12)
        
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)

class ForwardTimeShift(BaseWaveformTransform):
    """
       Forward shift of samples up to specified duration.  
       Parameters:
       p: Default=0.5
       max_duration: maximum audio length (in seconds), Default=10
       sr: sample rate, Default=16000
    
    """
    def __init__(self, always_apply=False, p=0.5, max_duration=10, sr=16000):

        super(ForwardTimeShift, self).__init__(always_apply, p)
        self.max_duration = max_duration
        self.sr = sr

    def apply(self, waveform, **params):
      waveform = waveform.clone()

      shift = np.random.randint(0, int(self.sr*self.max_duration - waveform.shape[-1]))
      waveform = torch.cat((torch.zeros(1, shift, dtype=torch.float), waveform), dim = 1)
      return waveform

class Inversion(BaseWaveformTransform):
    """
       Reverse signes of samples.
       
       Parameters:
       p: Default=0.5
    """
    def __init__(self, always_apply=False, p=0.5):

        super(Inversion, self).__init__(always_apply, p)

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        print('inverse')
        return -waveform         

class ZeroSamples(BaseWaveformTransform):
    """
       Set to zero some samples in waveform.
       
       
       Parameters:
       p: Default=0.5
       max_percent: maximal percentage of samples which will be set to zeros, Default=0.5
       min_percent: minimal percentage of samples which will be set to zeros, Default=0.0
    """
    def __init__(self, always_apply=False, p=0.5, min_percent=0.0, max_percent=0.5):

        super(ZeroSamples, self).__init__(always_apply, p)
        self.min_percent = min_percent
        self.max_percent = max_percent

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        percent = np.random.uniform(self.min_percent, self.max_percent)
        n_zero_samples = int(percent * waveform.shape[1])
        indexes = np.random.randint(0, waveform.shape[1], size=(n_zero_samples,))
        waveform.scatter_(1, torch.tensor(indexes).unsqueeze(0), torch.zeros((1,n_zero_samples), dtype=torch.float))
        return waveform

class ClippingSamples(BaseWaveformTransform):
    """
       Clip some samples from waveform.
       
       Parameters:
       p: Default=0.5
       max_percent: maximal percentage of samples which will be clipped, Default=0.25
       min_percent: minimal percentage of samples which will be clipped, Default=0.0
    """
    def __init__(self, always_apply=False, p=0.5, min_percent=0.0, max_percent=0.25):

        super(ClippingSamples, self).__init__(always_apply, p)
        self.min_percent = min_percent
        self.max_percent = max_percent

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        percent = np.random.uniform(self.min_percent, self.max_percent)
        n_clip_samples = int(percent * waveform.shape[1])
        indexes = np.random.randint(0, waveform.shape[1], size=(n_clip_samples,))
        waveform.scatter_(1, torch.tensor(indexes).unsqueeze(0), 10000000*torch.ones((1,n_clip_samples), dtype=torch.float))
        waveform = waveform[waveform != 10000000.].unsqueeze(0)
        return waveform


class ColoredNoise(BaseWaveformTransform):
    """ Add different types of noise to single-channel waveform

        noise_color -- noise type is randomly chosen:  0 -- white noise (flat power spectrum), 
                                                       1 -- pink noise (~1/f),
                                                       2 -- brown noise (~1/f^2),
                                                       -1 -- blue noise (~f),
                                                       -2 -- violet noise (~f^2)

                                                       3 -- grey noise (brown + violet noise)
                            
                            
       Parameters:
       p: Default=0.5
       max_amp: maximal percentage of noise amplitude in relation to waveform amplitude, Default=0.1
       min_amp: minimal percentage of noise amplitude in relation to waveform amplitude, Default=0.0
    """

    def __init__(self, always_apply=False, p=0.5, max_amp=0.1, min_amp=0.0):

        super(ColoredNoise, self).__init__(always_apply, p)
        self.min_amp = min_amp
        self.max_amp = max_amp


    def apply(self, waveform, **params):
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy() # calculate noise amplitude
        noise_color = np.random.randint(-2,3) # случайный тип шума
        if noise_color != 3:
            col_noise = colorednoise.powerlaw_psd_gaussian(noise_color, len(waveform)) # noise generation
        else:
            # grey noise
            col_noise = colorednoise.powerlaw_psd_gaussian(2, len(waveform)) + colorednoise.powerlaw_psd_gaussian(-2, len(waveform))
        
        col_noise = col_noise*noise_amp/np.abs(col_noise).max() # get noise with desired amplitude

        waveform = waveform + torch.tensor(col_noise)
        return waveform.unsqueeze(0).type(torch.float)