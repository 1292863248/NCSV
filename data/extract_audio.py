import os
import ffmpeg
import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish').cuda()
model.eval()
from tqdm import tqdm
with torch.no_grad():
    for file in tqdm(os.listdir('audio')):
        if file.split('.')[-1] =='wav':
            file_name = file.split('.')[0]
            fea = model.forward(f'audio_wav/{file}')
            torch.save(fea, os.path.join('audio_features', file_name + '.pt'))
