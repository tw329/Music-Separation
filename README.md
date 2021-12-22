# Music-Separation

## Training and loss
![trian and test loss](https://github.com/tw329/Music-Separation/blob/main/6F61A858-4FD1-4C86-AFF4-AA40A547C73A.jpeg)

+ **Train**
     
     ``python wav.py .\musdb18\ .\musdb18_wav\``
     
     ``python train.py .\musdb18_wav\ .\output\``
     
+ **Run**
     
     ``python test.py .\model.pth``
     

## Requirements
#### GPU with cuda
#### librosa
#### soundfile
#### pytorch
#### torchaudio
#### youtube-dl

## Reference
+ **https://github.com/f90/Wave-U-Net**
+ **https://github.com/ShichengChen/WaveUNet**
+ **https://github.com/alexeyrodriguez/waveunet**
+ **https://github.com/jnzhng/keras-unet-vocal-separation**
+ **https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer/blob/master/network.py**






