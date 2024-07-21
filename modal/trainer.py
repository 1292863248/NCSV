import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from .ncsv import NCSV
from .dataset import NCSVDataset, collect_fn
import copy
from .single_modal_baseline import *
def train(**kwargs):
    dataset_train=NCSVDataset(data_path=kwargs['data_path'],json_path=kwargs['train_json_path'],
                                        bert_path=kwargs['bert_path'],max_length=kwargs['bert_max_length'])
    dataset_test = NCSVDataset(data_path=kwargs['data_path'], json_path=kwargs['test_json_path'],
                                          bert_path=kwargs['bert_path'], max_length=kwargs['bert_max_length'])
    train_dataloader = DataLoader(dataset_train,batch_size=kwargs['batch_size'],
                                  num_workers=kwargs['num_workers'],
                                  shuffle=True,
                                  collate_fn=collect_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=kwargs['batch_size'],
                                  num_workers=kwargs['num_workers'],
                                  shuffle=True,
                                  collate_fn=collect_fn)
    criterion = nn.CrossEntropyLoss()
    if kwargs['model']=='NCSV':
        model=NCSV(bert_path=kwargs['bert_path'],hidden_dim=kwargs['hidden_dim'],
                    text_dim=kwargs['text_dim'],video_dim=kwargs['video_dim'],
                    audio_dim=kwargs['audio_dim'],dropout=kwargs['dropout'])
    elif kwargs['model']=='bert_base_chinese':
        model=bert_base_chinese(bert_path=kwargs['bert_path'],fea_dim=kwargs['hidden_dim'],text='social')
    elif kwargs['model']=='clip_text_encoder':
        model=clip_text_encoder(bert_path=kwargs['bert_path'],fea_dim=kwargs['hidden_dim'],text='social')
    elif kwargs['model']=='Vggish':
        model = Vggish(fea_dim=kwargs['hidden_dim'])
    elif kwargs['model']=='clip_image_encoder':
        model=clip_image_encoder(fea_dim=kwargs['hidden_dim'])
    elif kwargs['model'] == 'TimeSFormer':
        model=TimeSFormer(fea_dim=kwargs['hidden_dim'])
    elif kwargs['model'] == 'AST':
        model =AST(fea_dim=kwargs['hidden_dim'])
    elif kwargs['model'] == 'SVFENDModel':
        model=SVFENDModel()

    model=model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=kwargs['lr'])
    best_acc_test = 0.0
    best_metric={}
    for epoch in range(10):

        #train
        model.train()
        running_loss = 0.0
        tpred = []
        tlabel = []
        for batch in tqdm(train_dataloader):
            batch_data = batch
            for k, v in batch_data.items():
                batch_data[k] = v.cuda()
            label = batch_data['label']
            outputs = model(**batch_data)
            _,preds = torch.max(outputs, 1)  # 概率高的为预测结果
            loss = criterion(outputs, label)
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            tlabel.extend(label.detach().cpu().numpy().tolist())
            tpred.extend(preds.detach().cpu().numpy().tolist())
            running_loss += loss.item() * label.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(kwargs['folder'],f'train epoch:{epoch+1} ','Loss: {:.4f} '.format(epoch_loss))

        metrics = {}
        metrics['auc'] = roc_auc_score(tlabel, tpred)
        tpred = np.around(np.array(tpred)).astype(int)
        metrics['f1'] = f1_score(tlabel, tpred)
        metrics['recall'] = recall_score(tlabel, tpred)
        metrics['precision'] = precision_score(tlabel, tpred)
        metrics['acc'] = accuracy_score(tlabel, tpred)
        print(metrics)

        #test
        model.eval()
        tlabel=[]
        tpred=[]
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    batch_data[k] = v.cuda()
                label = batch_data['label']
                outputs = model(**batch_data)
                _, preds = torch.max(outputs, 1)
                tlabel.extend(label.detach().cpu().numpy().tolist())
                tpred.extend(preds.detach().cpu().numpy().tolist())

        metrics = {}
        metrics['auc'] = roc_auc_score(tlabel, tpred)
        tpred = np.around(np.array(tpred)).astype(int)
        metrics['f1'] = f1_score(tlabel, tpred)
        metrics['recall'] = recall_score(tlabel, tpred)
        metrics['precision'] = precision_score(tlabel, tpred)
        metrics['acc'] = accuracy_score(tlabel, tpred)
        print(kwargs['folder'],f'test epoch:{epoch+1} ', metrics)
        if metrics['acc']>best_acc_test:
            best_acc_test=metrics['acc']
            best_metric=metrics
            torch.save(model.state_dict(),
                       kwargs['save_param_path']+ '/'+kwargs['folder'] + "_{0:.4f}".format(best_acc_test))
            print('save model')

        if metrics['acc']< best_acc_test and best_acc_test !=0.0:
            print('stop')
            break

    return best_metric

