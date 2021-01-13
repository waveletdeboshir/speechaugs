# speechaugs
Augmentations for single-channel waveforms.
***
### Current augmentations:
* Time Stretch
* Pitch Shift
* Colored Noise (white, pink, brown, blue, violet, grey)
***
## Installation
`pip install speechaugs`
***
## Time Stretch
Stretch a wavefom with randomly chosen rate. Is implemented using <a href="https://librosa.org/doc/main/generated/librosa.effects.time_stretch.html"> *librosa.effects.time_stretch*</a>. 
<p>
<img src="images/timestretch.png" width="700" height="300"/> 
</p>

## Pitch Shift
Shift a pitch by *n_steps* semitones. Is implemented using <a href="https://librosa.org/doc/main/generated/librosa.effects.time_stretch.html"> *librosa.effects.time_stretch*</a>. 

The work of PitchShift can be better illustrated on the MelSpectrograms of waveforms. 

**Higher pitch (+9 semitones):**
<p>
<img src="images/higherpitch.png" width="600" height="300" title="Higher pitch (+9 semitones)"/> 
</p>

**Lower pitch (-5 semitones)**
<p>
<img src="images/lowerpitch.png" width="600" height="300" title="Lower pitch (-5 semitones)"/> 
</p>

## Colored Noise
Color of noise depends on the spectral density of the noise. You can go to <a href="https://en.wikipedia.org/wiki/Colors_of_noise">wiki page</a> for more information.

This class is written using <a href="https://github.com/felixpatzelt/colorednoise">colorednoise package</a>. The color of noise is randomly choosen.

**White Noise**
<p>
<img src="images/whitenoise.png" width="600" height="300" title="White Noise"/> 
</p>

**Brown Noise**
<p>
<img src="images/brownnoise.png" width="600" height="300" title="Brown Noise"/> 
</p>

***
## Usage example
Import:
```python
from speechaugs import TimeStretch, PitchShift, ColoredNoise
    
import torch, torchaudio
import albumentations as A
```
Usage:
```python

ex_waveform, sr = torchaudio.load('audio_filename')

transforms = A.Compose([
    TimeStretch(p=0.5, sr=sr),
    PitchShift(p=0.5, sr=sr),
    ColoredNoise(p=0.5)
], p=1.0)

augmented = transforms(waveform=ex_waveform)['waveform']
```