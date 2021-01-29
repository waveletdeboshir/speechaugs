import torch, torchaudio
from librosa.effects import time_stretch, pitch_shift
from colorednoise import powerlaw_psd_gaussian
import numpy as np
import glob

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
        p: float, Default=0.5
        max_duration: float, maximum audio length (in seconds), Default=10.
        sr: int, sample rate, Default=16000
        min_rate: float, minimal stretch rate, Default=0.5
        max_rate: float, maximal stretch rate, Default=2.
        
       -------------
       input: waveform, torch.tensor of shape [1, n1]   
       output: augmented waveform, torch.tensor of shape [1, n2]  
    """

    def __init__(self, always_apply=False, p=0.5, sr=16000, max_duration=10., min_rate=0.5, max_rate=2.):

        super(TimeStretchLibrosa, self).__init__(always_apply, p)
        assert min_rate < max_rate, 'min_rate >= max_rate'

        self.max_duration = max_duration
        self.sr = sr
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply(self, waveform, **params):
        assert waveform.shape[1] <= self.max_duration*self.sr, 'waveform length > max_duration*sr'
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        
        rate = np.random.uniform(self.min_rate, self.max_rate) # rate < 1.0 -- slow down, rate > 1.0 -- speed up

        if waveform.shape[1]/rate>=self.max_duration*self.sr-1000:
            rate = np.random.uniform(1., 2.) # If length is greater than max_duration then we increase speed up to 2 times
        waveform = time_stretch(waveform[0].numpy(), rate)
                
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)

class PitchShiftLibrosa(BaseWaveformTransform):
    """ 
        Shift a pitch up or down by n semitones. 
        
        
        Parameters:
        p: float, Default=0.5
        sr: int, sample rate, Default=16000    
        min_steps: int, minimal shift steps (semitones), Default = -10
        max_steps: int, maximal shift steps (semitones), Default = 10
        
       -------------
       input: waveform, torch.tensor of shape [1, n]   
       output: augmented waveform, torch.tensor of shape [1,n]  
    """
    def __init__(self, always_apply=False, p=0.5, sr=16000, min_steps=-10, max_steps=10):

        super(PitchShiftLibrosa, self).__init__(always_apply, p)
        assert min_steps < max_steps, 'min_steps >= max_steps'

        self.sr = sr
        self.min_steps = min_steps
        self.max_steps = max_steps


    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()

        n_steps = np.random.randint(self.min_steps, self.max_steps) # n_steps < 0 -- shift down, n_steps > 0 -- shift up
        waveform = pitch_shift(waveform[0].numpy(), self.sr, n_steps=n_steps, bins_per_octave=12)
        
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)

class ForwardTimeShift(BaseWaveformTransform):
    """
       Forward shift of samples up to specified duration.  
       
       Parameters:
       p: float, Default=0.5
       max_duration: float, maximum audio length (in seconds), Default=10.
       sr: int, sample rate, Default=16000
       
       -------------
       input: waveform, torch.tensor of shape [m, n1]   
       output: augmented waveform, torch.tensor of shape [m, n2] (n2>n1)  
    """
    def __init__(self, always_apply=False, p=0.5, max_duration=10., sr=16000):

        super(ForwardTimeShift, self).__init__(always_apply, p)
        self.max_duration = max_duration
        self.sr = sr

    def apply(self, waveform, **params):
      assert waveform.shape[1] <= self.max_duration*self.sr, 'waveform length > max_duration*sr'
      assert waveform.shape[1] > 0, 'waveform is empty'
      waveform = waveform.clone()
      dif = int(self.sr*self.max_duration - waveform.shape[1])
      if dif > 0:
          shift = np.random.randint(0, dif)
          waveform = torch.cat((torch.zeros(waveform.shape[0], shift, dtype=torch.float), waveform), dim = 1)
          return waveform
      else:
          return waveform # if waveform length == max_duration*sr then we don't shift

class Inversion(BaseWaveformTransform):
    """
       Reverse signes of samples.
       
       Parameters:
       p: float, Default=0.5
       
       -------------
       input: waveform, torch.tensor of shape [m, n]   
       output: augmented waveform, torch.tensor of shape [m, n]  
    """
    def __init__(self, always_apply=False, p=0.5):

        super(Inversion, self).__init__(always_apply, p)

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        return -waveform         

class ZeroSamples(BaseWaveformTransform):
    """
       Set to zero some samples in waveform.
       
       
       Parameters:
       p: float, Default=0.5
       max_percent: float, maximal percentage of samples which will be set to zeros, Default=0.5
       min_percent: float, minimal percentage of samples which will be set to zeros, Default=0.0
       
       -------------
       input: waveform, torch.tensor of shape [1, n]   
       output: augmented waveform, torch.tensor of shape [1, n]  
    """
    def __init__(self, always_apply=False, p=0.5, min_percent=0.0, max_percent=0.5):

        super(ZeroSamples, self).__init__(always_apply, p)
        assert min_percent < max_percent, 'min_percent >= max_percent'
        self.min_percent = min_percent
        self.max_percent = max_percent

    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        percent = np.random.uniform(self.min_percent, self.max_percent)
        n_zero_samples = int(percent * waveform.shape[1])
        indexes = np.random.randint(0, waveform.shape[1], size=(n_zero_samples,))
        waveform.scatter_(1, torch.tensor(indexes).unsqueeze(0), torch.zeros((1,n_zero_samples), dtype=torch.float))
        return waveform

class ClippingSamples(BaseWaveformTransform):
    """
       Clip some samples from a single-channel waveform.
       
       Parameters:
       p: float, Default=0.5
       max_percent: float, maximal percentage of samples which will be clipped, Default=0.25
       min_percent: float, minimal percentage of samples which will be clipped, Default=0.0
       
       -------------
       input: waveform, torch.tensor of shape [1, n1]   
       output: augmented waveform, torch.tensor of shape [1, n2] (n2<n1)  
    """
    def __init__(self, always_apply=False, p=0.5, min_percent=0.0, max_percent=0.25):

        super(ClippingSamples, self).__init__(always_apply, p)
        assert min_percent < max_percent, 'min_percent >= max_percent'
        self.min_percent = min_percent
        self.max_percent = max_percent

    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
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
       p: float, Default=0.5
       max_amp: float, maximal percentage of noise amplitude in relation to waveform amplitude, Default=0.1
       min_amp: float minimal percentage of noise amplitude in relation to waveform amplitude, Default=0.0
       
       -------------
       input: waveform, torch.tensor of shape [1, n]   
       output: augmented waveform, torch.tensor of shape [1, n]  
    """

    def __init__(self, always_apply=False, p=0.5, max_amp=0.1, min_amp=0.0):

        super(ColoredNoise, self).__init__(always_apply, p)
        assert min_amp < max_amp, 'min_amp >= max_amp'
        self.min_amp = min_amp
        self.max_amp = max_amp


    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy() # calculate noise amplitude
        noise_color = np.random.randint(-2, 4) # noise type
        if noise_color != 3:
            col_noise = powerlaw_psd_gaussian(noise_color, len(waveform)) # noise generation
        else:
            # grey noise
            col_noise = powerlaw_psd_gaussian(2, len(waveform)) + powerlaw_psd_gaussian(-2, len(waveform))
        
        col_noise = col_noise*noise_amp/np.abs(col_noise).max() # get noise with desired amplitude

        waveform = waveform + torch.tensor(col_noise)
        return waveform.unsqueeze(0).type(torch.float)
    
    
class LoudnessChange(BaseWaveformTransform):
    """
       Change loudness of the intervals of a waveform.

       Parameters:
       p: float, Default=0.5
       min_factor: float,0.0<min_factor<1.0 minimal factor for reducing loudness (i.e. factor = 0.5 means reducing loudness by half), Default=0.1
       max_factor: float, >1.0 maximal factor for increasing loudness, Default=10.
       max_n_intervals: int, maximal number of intervals for changing loudness, Default=5   

       -------------
       input: waveform, torch.tensor of shape [m, n]   
       output: augmented waveform, torch.tensor of shape [m, n]  
    """

    def __init__(self, always_apply=False, p=0.5, min_factor=0.1, max_factor=10., max_n_intervals=5):

        super(LoudnessChange, self).__init__(always_apply, p)
        assert 0.< min_factor <1., 'min_factor >= 1. or <=0.'
        assert max_factor >1., 'max_factor <= 1.'
        assert max_n_intervals > 0, 'max_n_intervals <= 0'
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.max_n_intervals = max_n_intervals

    def apply(self, waveform, **params):
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        n_intervals = np.random.randint(1, self.max_n_intervals+1)
        
        len_interval = len(waveform[0])//n_intervals
        for i in range(n_intervals):

            if np.random.binomial(1, 0.5):
                factor = np.random.uniform(self.min_factor, 1.) # reduce loudness
            else:
                factor = np.random.uniform(1., self.max_factor) # increase loudness
            waveform[:, i*len_interval:(i+1)*len_interval] = factor*waveform[:, i*len_interval:(i+1)*len_interval]

        return waveform

class ShortNoises(BaseWaveformTransform):
    """
       Add short colored noises to a single-channel waveform.

       noise_color -- noise type is randomly chosen:  0 -- white noise (flat power spectrum), 
                                                       1 -- pink noise (~1/f),
                                                       2 -- brown noise (~1/f^2),
                                                       -1 -- blue noise (~f),
                                                       -2 -- violet noise (~f^2)
                                                       3 -- grey noise (brown + violet noise)

       Parameters:
       p: float, Default=0.5
       max_amp: float, maximal percentage of noise amplitude in relation to waveform amplitude, Default=0.5
       min_amp: float, minimal percentage of noise amplitude in relation to waveform amplitude, Default=0.3
       max_n_noises: int, maximal number of short noises in a waveform, Default=5
       
       -------------
       input: waveform, torch.tensor of shape [1, n]   
       output: augmented waveform, torch.tensor of shape [1, n] 
    """
    def __init__(self, always_apply=False, p=0.5, min_amp=0.3, max_amp=0.5, max_n_noises=5):
        super(ShortNoises, self).__init__(always_apply, p)
        assert min_amp < max_amp, 'min_amp >= max_amp'
        assert 0 < max_n_noises < 50, 'max_n_intervals <= 0 or >= 50'
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.max_n_noises = max_n_noises

    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        waveform.squeeze_(0)
        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy() # calculate noise amplitude
        noise_color = np.random.randint(-2, 4) # noise type
        n_noises = np.random.randint(1, self.max_n_noises+1)
        for i in range(n_noises):
            noise_length = int(np.random.uniform(0.01, 1/(self.max_n_noises*2.))*len(waveform)) 
            start_sample = np.random.randint(i*(len(waveform)//n_noises), (i+1)*(len(waveform)//n_noises) - noise_length)
            if noise_color != 3:
                col_noise = powerlaw_psd_gaussian(noise_color, noise_length) # noise generation
            else:
                # grey noise
                col_noise = powerlaw_psd_gaussian(2, noise_length) + powerlaw_psd_gaussian(-2, noise_length)
            
            col_noise = col_noise*noise_amp/np.abs(col_noise).max() # get noise with desired amplitude
            waveform[start_sample:start_sample+noise_length] = waveform[start_sample:start_sample+noise_length]+col_noise
        return waveform.unsqueeze(0)


class FileNoise(BaseWaveformTransform):
    """
       Add noise to a single-channel waveform from randomly chosen file.

       Parameters:
       root_path: str, path to the folder with noise files.
       p: float, Default=0.5
       max_amp: float, maximal percentage of noise amplitude in relation to waveform amplitude, Default=0.5
       min_amp: float, minimal percentage of noise amplitude in relation to waveform amplitude, Default=0.3  
       sr: int, sample rate, Default=16000

       -------------
       input: waveform, torch.tensor of shape [1, n]   
       output: augmented waveform,torch.tensor of shape [1, n]  
    """
    def __init__(self, root_path, always_apply=False, p=0.5, min_amp=0.1, max_amp=0.5, sr=16000):
        super(FileNoise, self).__init__(always_apply, p)
        assert min_amp < max_amp, 'min_amp >= max_amp'
        self.root_path = root_path
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.sr = sr

    def get_noise_paths(self, root):
        
        extens = ('.wav', '.mp3', '.flac', '.ogg', '.opus')
        noise_paths = []
        paths = glob.glob(root + '**', recursive=True)
        for filepath in paths:
            if filepath.endswith(extens):
                noise_paths.append(filepath)

        return noise_paths
       

    def apply(self, waveform, **params):
        assert waveform.shape[0] == 1, 'waveform should have 1-channel'
        assert waveform.shape[1] > 0, 'waveform is empty'
        waveform = waveform.clone()
        wave_len = len(waveform[0])
        noise_files = self.get_noise_paths(self.root_path)
        ind = np.random.randint(len(noise_files))
        noise_wave, sr_noise = torchaudio.load(noise_files[ind])
        assert sr_noise == self.sr, "Sample rates of two waveforms are different. Please, change sr of your noisy files to "+str(self.sr)
        if noise_wave.shape[0] > 1:
            noise_wave = noise_wave.float().mean(dim=0, keepdim=True)
        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy()
        noise_wave = noise_wave*noise_amp/noise_wave.abs().max()
        noise_len = len(noise_wave[0])
        if noise_len >= wave_len:
            start_ind = np.random.randint(noise_len - wave_len)
            waveform = waveform + noise_wave[:, start_ind:start_ind + wave_len]
        else:
            n_rep = wave_len//noise_len
            for i in range(n_rep):
                waveform[:, i*noise_len:(i+1)*noise_len] = waveform[:, i*noise_len:(i+1)*noise_len] + noise_wave
            waveform[:, n_rep*noise_len:] = waveform[:, n_rep*noise_len:] + noise_wave[:, :wave_len - n_rep*noise_len]

        return waveform
