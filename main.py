import argparse
import os.path
import random

import numpy as np

from modal import trainer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='NCSV')
parser.add_argument('--data_path',type=str,default='./data')
parser.add_argument('--train_json_path',type=str,default='train_data.json')
parser.add_argument('--test_json_path',type=str,default='test_data.json')
parser.add_argument('--cross_validation_json_path',type=str,default='cross-validation-data')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epoch_stop', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--save_param_path', default= './checkpoint')
parser.add_argument('--path_tensorboard', default= './tb/')
parser.add_argument('--bert_path', default= './AltClip')
parser.add_argument('--bert_max_length', default= 512)
parser.add_argument('--hidden_dim', default= 512)
parser.add_argument('--text_dim', default= 768)
parser.add_argument('--video_dim', default= 768)
parser.add_argument('--audio_dim', default= 128)
parser.add_argument('--use_sentiment', default= False)
parser.add_argument('--sentiment_dim',default=0)
parser.add_argument('--fold_validation', default=False)
args = parser.parse_args()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
config = {
    'model':args.model,
    'data_path':args.data_path,
    'train_json_path':args.train_json_path,
    'test_json_path':args.test_json_path,
    'epoches': args.epoches,
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'epoch_stop': args.epoch_stop,
    'seed': args.seed,
    'device': args.gpu,
    'lr': args.lr,
    'dropout': args.dropout,
    'weight_decay': args.weight_decay,
    'save_param_path': args.save_param_path,
    'path_tensorboard': args.path_tensorboard,
    'bert_path':args.bert_path,
    'bert_max_length':args.bert_max_length,
    'hidden_dim':args.hidden_dim,
    'text_dim':args.text_dim,
    'video_dim':args.video_dim,
    'audio_dim':args.audio_dim
        }

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
folders=['folder1','folder2','folder3','folder4']

acc=[]
precision=[]
recall=[]
f1=[]

for folder in folders:
    print('start '+folder)
    config['test_json_path']=os.path.join(args.cross_validation_json_path,f'{folder}.json')
    config['train_json_path'] = os.path.join(args.cross_validation_json_path, f'without_{folder}.json')
    config['folder']=folder
    best_metric=trainer.train(**config)
    print(folder,best_metric)
    acc.append(best_metric['acc'])
    precision.append(best_metric['precision'])
    recall.append(best_metric['recall'])
    f1.append(best_metric['f1'])


print('acc:',np.mean(acc),'std:',np.std(acc))
print('precision:',np.mean(precision),'std:',np.std(precision))
print('recall:',np.mean(recall),'std:',np.std(recall))
print('f1:',np.mean(f1),'std:',np.std(f1))


