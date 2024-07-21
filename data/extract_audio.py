import os
import ffmpeg
import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish').cuda()
model.eval()
from tqdm import tqdm
done_list=[]
for file in os.listdir('audio_features'):
    done_list.append(file.split('.')[0])
with torch.no_grad():
    for file in tqdm(os.listdir('audio_wav')):
        if file.split('.')[0] not in done_list:
            file_name = file.split('.')[0]
            fea = model.forward(f'audio_wav/{file}')
            torch.save(fea, os.path.join('audio_features', file_name + '.pt'))









