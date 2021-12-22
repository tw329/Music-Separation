import os
import sys
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf



origin_dataset_dir = sys.argv[1]
new_dataset_dir = sys.argv[2]
sr = 22050

os.mkdir(new_dataset_dir)

os.mkdir(os.path.join(new_dataset_dir, 'train'))
os.mkdir(os.path.join(new_dataset_dir, 'test'))

with tempfile.TemporaryDirectory() as tmpdir:
    for subdir in ('train', 'test'):
        origin_dir = os.path.join(origin_dataset_dir, subdir)
        files = [f for f in os.listdir(origin_dir)
                    if os.path.splitext(f)[1] == '.mp4']
        for file in files:
            path = os.path.join(origin_dir, file)
            name = os.path.splitext(file)[0]
            wav_data = []

            for channel in range(5):
                temp_fn = name+"."+str(channel)+".wav"
                out_path = os.path.join(tmpdir, temp_fn)
                subprocess.run(['ffmpeg', '-i', path, '-map', "0:"+str(channel), out_path])
                sound, _ = librosa.load(out_path, sr=sr, mono=True)
                wav_data.append(sound)
            wav_data = np.stack(wav_data, axis=1)
            out_path = os.path.join(new_dataset_dir, subdir, name+".wav")
            sf.write(out_path, wav_data, sr)