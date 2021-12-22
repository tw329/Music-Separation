# Music-Separation

## Training and loss
![trian and test loss](https://user-images.githubusercontent.com/13598741/117584790-dfdc9100-b0dc-11eb-87a6-521ac43c1189.PNG)

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






