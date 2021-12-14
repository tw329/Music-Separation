import os
import sys
import subprocess
import tempfile
import random

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchaudio

def read_data(dataset_dir, n_fft, data_len, sampling_rate):
    window = torch.hann_window(n_fft)
    with torch.no_grad():

        train_dir = os.path.join(dataset_dir, 'train')
        wav_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                     if os.path.splitext(f)[1] == '.wav']
        train = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            sound_spec = torch.stft(sound, n_fft, window=window)
            sound_spec = sound_spec.pow(2).sum(-1).sqrt()
            x = sound_spec[0]
            t = sound_spec[1:]

            hop = data_len // 4

            for n in range((x.size(1) - data_len) // hop + 1):
                start = n * hop
                train.append((x[:, start:start + data_len],
                              t[:, :, start:start + data_len]))


        test_dir = os.path.join(dataset_dir, 'test')
        wav_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if os.path.splitext(f)[1] == '.wav']
        test = []
        for path in wav_files:
            sound, sr = torchaudio.load(path)
            assert sr == sampling_rate
            split_sound = torch.split(sound, sound.size(1) // 2, dim=1)
            test.extend(split_sound)

    return train, test

class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, data, out_len):
        self.data = data
        self.len = out_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, t = self.data[item]
        length = x.size(1)
        i = random.randint(0, length - self.len)
        return x[:, i:i + self.len], t[:, :, i:i + self.len]


def train(model, data_loader, optimizer, device, epoch, loss_record):
    model.train()

    train_loss = 0
    for x, t in data_loader:
        batch_size = x.size(0)
        x, t = x.to(device), t.to(device)
        y = model(x) * x.unsqueeze(1)

        loss = F.l1_loss(y, t, reduction='sum') / batch_size
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_record.add_scalar('train/loss', train_loss / len(data_loader), epoch)

def test(model, test_data, device, epoch, loss_record):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        window = torch.hann_window(2047, device=device)
        for sound in test_data:
            sound = sound.to(device)
            sound_stft = torch.stft(sound, 2047, window=window)
            sound_spec = sound_stft.pow(2).sum(-1).sqrt()
            x, t = sound_spec[0], sound_spec[1:]

            x_padded, (left, right) = padding(x)
            right = x_padded.size(1) - right
            mask = model(x_padded.unsqueeze(0)).squeeze(0)[:, :, left:right]
            y = mask * x.unsqueeze(0)
            loss = F.l1_loss(y, t, reduction='sum')
            test_loss += loss.item()
    loss_record.add_scalar('test/loss', test_loss / len(test_data), epoch)

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
    n_frames = sound_stft.size(-1)
    n_pad = (64 - n_frames % 64) % 64
    if n_pad:
        left = n_pad // 2
        right = n_pad - left
        return F.pad(sound_stft, (left, right)), (left, right)
    else:
        return sound_stft, (0, 0)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend("soundfile")
    wav_dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    learning_rate = 0.01
    batch_size = 32
    epochs = 500

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    model = Unet(4)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_data, test_data = read_data(wav_dataset_dir, 2047, 512, 22050)
    train_dataset = RandomDataset(train_data, 256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=False)

    loss_record = SummaryWriter()

    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, device, epoch, loss_record)
        print("\n\ntrain", epoch, "\n\n")
        if epoch % 25 == 0:
            test(model, test_data, device, epoch, loss_record)
            model.cpu()
            state_dict = model.state_dict()
            path = os.path.join(output_dir, "model-"+str(epoch)+".pth")
            torch.save(state_dict, path)
            model.to(device)
            print("\n\ntest", epoch, "\n\n")

    loss_record.close()
    
if __name__ == '__main__':
    main()