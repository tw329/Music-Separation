import sys
import os

from mir_eval.separation import bss_eval_sources
import soundfile as sf
import torch
import torchaudio

import torch.nn.functional as F

class Unet(torch.nn.Module):

    def __init__(self, n_class):
        super(Unet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv_bn1 = torch.nn.BatchNorm2d(16)        
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)        
        self.conv_bn2 = torch.nn.BatchNorm2d(32)        
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)        
        self.conv_bn3 = torch.nn.BatchNorm2d(64)        
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)        
        self.conv_bn4 = torch.nn.BatchNorm2d(128)        
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)        
        self.conv_bn5 = torch.nn.BatchNorm2d(256)        
        self.conv6 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.conv_bn6 = torch.nn.BatchNorm2d(512)
    #----------------------------------------------------------------------------------------------------------------
        self.deconv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)        
        self.deconv_bn1 = torch.nn.BatchNorm2d(256)        
        self.dropout1 = torch.nn.Dropout2d(0.2)        
        self.deconv2 = torch.nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1)        
        self.deconv_bn2 = torch.nn.BatchNorm2d(128)        
        self.dropout2 = torch.nn.Dropout2d(0.2)        
        self.deconv3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)        
        self.deconv_bn3 = torch.nn.BatchNorm2d(64)        
        self.dropout3 = torch.nn.Dropout2d(0.2)        
        self.deconv4 = torch.nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)        
        self.deconv_bn4 = torch.nn.BatchNorm2d(32)        
        self.deconv5 = torch.nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2, output_padding=1)        
        self.deconv_bn5 = torch.nn.BatchNorm2d(16)       
        self.deconv6 = torch.nn.ConvTranspose2d(32, 4, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = torch.log(x + 1e-8)
        h1 = F.leaky_relu(self.conv_bn1(self.conv1(x)), 0.2)
        h2 = F.leaky_relu(self.conv_bn2(self.conv2(h1)), 0.2)
        h3 = F.leaky_relu(self.conv_bn3(self.conv3(h2)), 0.2)
        h4 = F.leaky_relu(self.conv_bn4(self.conv4(h3)), 0.2)
        h5 = F.leaky_relu(self.conv_bn5(self.conv5(h4)), 0.2)
        h = F.leaky_relu(self.conv_bn6(self.conv6(h5)), 0.2)

        h = self.dropout1(F.relu(self.deconv_bn1(self.deconv1(h))))
        h = torch.cat((h, h5), dim=1)
        h = self.dropout2(F.relu(self.deconv_bn2(self.deconv2(h))))
        h = torch.cat((h, h4), dim=1)
        h = self.dropout3(F.relu(self.deconv_bn3(self.deconv3(h))))
        h = torch.cat((h, h3), dim=1)
        h = F.relu(self.deconv_bn4(self.deconv4(h)))
        h = torch.cat((h, h2), dim=1)
        h = F.relu(self.deconv_bn5(self.deconv5(h)))
        h = torch.cat((h, h1), dim=1)
        h = F.softmax(self.deconv6(h), dim=1)
        return h



def padding(sound_stft):
    frames = sound_stft.size(-1)
    pad = (64 - frames % 64) % 64
    if pad != 0:
        l = pad // 2
        r = pad - l
        return F.pad(sound_stft, (l, r)), (l, r)
    else:
        return sound_stft, (0, 0)


torch.cuda.empty_cache()
def main():
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")
    os.mkdir("input_music")
    URL = str(input("Youtube link:"))
    os.system("youtube-dl -x --audio-format wav -o \"./input_music/%(title)s.%(ext)s\" \""+URL+"\"")
    filename = os.listdir("input_music")[0]
    input_wav = "input_music/"+filename
    os.mkdir("output_music")
    output_wav = "./output_music/instrument_"+filename
    model_path = sys.argv[1]

    device = torch.device('cpu')

    with torch.no_grad():
        sound, _ = torchaudio.load(input_wav)
        sound = sound[[0], :].to(device)

        window = torch.hann_window(2047, device=device)

        sound_stft = torch.stft(sound, 2047, window=window)
        sound_spec = sound_stft.pow(2).sum(-1).sqrt()
        sound_spec, (left, right) = padding(sound_spec)

        model = Unet(4)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        right = sound_spec.size(2) - right
        mask = model(sound_spec).squeeze(0)[:, :, left:right]
        result = mask.unsqueeze(3) * sound_stft
        result = torch.istft(result, 2047, window=window, length=sound.size(-1))
        result = result.cpu().numpy()
    
    sf.write(output_wav, result.T, 44100)
    '''
    x, _ = sf.read(input_wav)
    y, _ = sf.read(output_wav)
    y = y[:,[0,0]]
    sdr = 0
    length = 0
    for i in range(0, len(x)):
        if x[i][0] != 0:
            sdr += bss_eval_sources(x[i], y[i])
            length += 1
    print(sdr/length)'''


if __name__ == '__main__':
    main()

print("Convert Successfully")