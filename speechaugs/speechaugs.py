import torch
import librosa
import colorednoise
import numpy as np

import albumentations as A
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



class TimeStretch(BaseWaveformTransform):
    """ Ускорение или замедление одноканальной waveform 
        max_duration -- заданная максимальная длина сигнала (в секундах)
        sr -- sample rate

        waveform -- torch.tensor shape=[n_channels, n]
    """

    def __init__(self, always_apply=False, p=0.5, sr=16000, max_duration=10):
        super(TimeStretch, self).__init__(always_apply, p)

        self.max_duration = max_duration
        self.sr = sr

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        
        rate = np.random.uniform(0.5, 2.0) # rate < 1.0 -- замедление, rate > 1.0 -- ускорение

        if waveform.shape[1]/rate>self.max_duration*self.sr:
            rate = np.random.uniform(1.0, 2.0) # Если длина получилась больше заданной максимальной -- ускоряем
        # print('rate=',rate)
        waveform = librosa.effects.time_stretch(waveform[0].numpy(), rate) # -- использует Griffin-Lim => будут шумы
        # waveform = pyrb.time_stretch(waveform.transpose(0,1).numpy(), self.sr, rate) # должно работать качественнее, но не устанавливается
                
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)

class PitchShift(BaseWaveformTransform):
    """ Сдвиг центральной частоты одноканальной waveform
        sr -- sample rate
        
    """
    def __init__(self, always_apply=False, p=0.5, sr=16000):
        super(PitchShift, self).__init__(always_apply, p)

        self.sr = sr


    def apply(self, waveform, **params):
        waveform = waveform.clone()

        n_steps = np.random.randint(-10, 10) # n_steps < 0 -- понижение частоты, n_steps > 0 -- повышение частоты
        # print('n_steps =', n_steps)
        waveform = librosa.effects.pitch_shift(waveform[0].numpy(), self.sr, n_steps=n_steps, bins_per_octave=12) #-- использует Griffin-Lim => будут шумы
        # сдвиг частоты на n_steps полутонов
        # waveform = pyrb.pitch_shift(waveform.transpose(0,1).numpy(), self.sr, n_steps=n_steps) # должно работать качественнее, но не устанавливается
        
        return torch.tensor(waveform, dtype=torch.float).unsqueeze(0)


class WhiteNoise(BaseWaveformTransform):
    """Добавление белого шума к одноканальной waveform
       
    """
    def __init__(self, always_apply=False, p=0.5, min_amp=0.0, max_amp=0.1):
        super(WhiteNoise, self).__init__(always_apply, p)
        self.min_amp = min_amp
        self.max_amp = max_amp

    def apply(self, waveform, **params):
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy()
        white_noise = np.random.randn(len(waveform))
        white_noise = white_noise*noise_amp/np.abs(white_noise).max()

        waveform = waveform + torch.tensor(white_noise)
        return waveform.unsqueeze(0).type(torch.float)



class PinkNoise(BaseWaveformTransform):
    '''Добавление розового шума к одноканальной waveform'''
    def __init__(self, always_apply=False, p=0.5, min_amp=0.0, max_amp=0.1):
        super(PinkNoise, self).__init__(always_apply, p)
        self.min_amp = min_amp
        self.max_amp = max_amp


    def apply(self, waveform, **params):
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy()
        
        pink_noise = colorednoise.powerlaw_psd_gaussian(1, len(waveform))
        pink_noise = pink_noise*noise_amp/np.abs(pink_noise).max()

        waveform = waveform + torch.tensor(pink_noise)
        return waveform.unsqueeze(0).type(torch.float)
    


class BrownNoise(BaseWaveformTransform):
    '''Добавление коричневого шума к одноканальной waveform'''
    def __init__(self, always_apply=False, p=0.5, min_amp=0.0, max_amp=0.1):
        super(BrownNoise, self).__init__(always_apply, p)
        self.min_amp = min_amp
        self.max_amp = max_amp


    def apply(self, waveform, **params):
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy()
        brown_noise = colorednoise.powerlaw_psd_gaussian(2, len(waveform))
        brown_noise = brown_noise*noise_amp/np.abs(brown_noise).max()

        waveform = waveform + torch.tensor(brown_noise)
        return waveform.unsqueeze(0).type(torch.float)


class ColoredNoise(BaseWaveformTransform):
    """ Добавление разных типов шума к одноканальной waveform
        min_amp -- мининмальная амплитуда шума в процентах (десятичных долях) от максимальной амплитуды исходной waveform
        max_amp -- максимальная амплитуда шума в процентах (десятичных долях) от максимальной амплитуды исходной waveform

        Случайно генерируется noise_color -- тип шума:  0 -- white noise (flat power spectrum), 
                                                        1 -- pink noise (~1/f),
                                                        2 -- brown noise (~1/f^2),
                                                        -1 -- blue noise (~f),
                                                        -2 -- violet noise (~f^2)

                                                        3 -- grey noise (brown + violet noise)
    """

    def __init__(self, always_apply=False, p=0.5, min_amp=0.0, max_amp=0.1):
        super(ColoredNoise, self).__init__(always_apply, p)
        self.min_amp = min_amp
        self.max_amp = max_amp


    def apply(self, waveform, **params):
        waveform = waveform.clone()
        waveform.squeeze_(0)

        noise_amp = np.random.uniform(self.min_amp, self.max_amp)*waveform.abs().max().numpy() # вычисление амплитуды шума
        noise_color = np.random.randint(-2,3) # случайный тип шума
        if noise_color != 3:
            col_noise = colorednoise.powerlaw_psd_gaussian(noise_color, len(waveform)) # генерация шума
        else:
            # генерация серого шума
            col_noise = colorednoise.powerlaw_psd_gaussian(2, len(waveform)) + colorednoise.powerlaw_psd_gaussian(-2, len(waveform))
        # print(noise_color)
        
        col_noise = col_noise*noise_amp/np.abs(col_noise).max() # приведение к нужной амплитуде

        waveform = waveform + torch.tensor(col_noise, dtype=torch.float)
        return waveform.unsqueeze(0).type(torch.float)
