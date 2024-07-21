import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from .coattention import *
from transformers import BertModel
from AltClip import AltCLIP
from .gnn_layer import *
from .temporal import temporal_encoder


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_head=3, dropout=0.1):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_dim, out_dim, dropout=dropout, alpha=0.1) for _ in
                           range(n_head)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(out_dim * n_head, out_dim, dropout=dropout, alpha=0.1)

    def forward(self, embedding, original_adj):
        # embedding:B,N,F
        # cosmatrix=calculate_similarity(embedding)
        # potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix), torch.zeros_like(cosmatrix))
        adj = original_adj  # + potentinal_adj
        adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj)).to(torch.float32)
        x = torch.cat([att(embedding, adj) for att in self.attentions], dim=2)  # (batch,num,hidden_dim*n)
        x = F.dropout(x, self.dropout)
        x = self.out_att(x, adj)  # (batch,num,hidden_dim)
        return x


class NCSV(nn.Module):
    def __init__(self, sentiment_dim=128, hidden_dim=512, text_dim=768, bert_path='../bert-base-chinese',
                 video_dim=768, audio_dim=128, dropout=0.1):
        super(NCSV, self).__init__()
        self.dim = hidden_dim
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.dropout = dropout
        self.bert_path = bert_path
        self.num_frames = 64
        self.num_audioframes = 64
        self.num_heads = 4
        if 'bert' in bert_path:
            self.bert = BertModel.from_pretrained(bert_path).requires_grad_(False)
            self.bert_last_hidden_dim = 768  # bert的last_hidden_state维度是768
        if 'AltClip' in bert_path:
            self.bert = getattr(AltCLIP, "AltCLIP").from_pretrain(
                download_path=f"{bert_path}/checkpoints",
                model_name="altclip-xlmr-l",
                only_download_config=False,
                device="gpu", ).requires_grad_(False)
            self.bert_last_hidden_dim = 1024  # Altclip的last_hidden_state维度是1024

        self.temporal_encoder = temporal_encoder(in_dim=self.video_dim + self.audio_dim, hidden_dim=hidden_dim,
                                                 out_dim=self.video_dim + self.audio_dim)
        self.co_attention_tv = co_attention(d_k=hidden_dim, d_v=hidden_dim, n_heads=self.num_heads,
                                            dropout=self.dropout,
                                            d_model=hidden_dim,
                                            visual_len=self.num_frames, sen_len=512,
                                            fea_v=self.video_dim + self.audio_dim, fea_s=self.bert_last_hidden_dim,
                                            pos=False)
        self.W1 = nn.Linear(sentiment_dim, text_dim)
        self.title_trans_fusion_param = nn.Parameter(torch.tensor(0.5))
        self.GAT = GAT(in_dim=text_dim, out_dim=hidden_dim, n_head=3, dropout=dropout)
        self.trm = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, **kwargs):

        ### Audio Frames ###
        audio_features = kwargs['audio_features']
        # audio_mask = kwargs['mask']

        ### Image Frames ###
        video_features = kwargs['video_features']
        video_mask = kwargs['video_mask']
        ### video text(ASR,OCR...)###
        # 取token级的特征
        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']
        transcript_inputid = kwargs['transcript_inputid']
        transcript_mask = kwargs['transcript_mask']
        if 'bert' in self.bert_path:
            title_features = self.bert(title_inputid, attention_mask=title_mask)[
                'last_hidden_state']  # （batch,length,dim=768)
            transcript_features = self.bert(transcript_inputid, attention_mask=transcript_mask)['last_hidden_state']
        if 'AltClip' in self.bert_path:
            _, title_features = self.bert.get_text_features(title_inputid,
                                                            attention_mask=title_mask)  # （batch,length,dim=1024)
            _, transcript_features = self.bert.get_text_features(transcript_mask, attention_mask=transcript_mask)
        text_features = title_features * self.title_trans_fusion_param + transcript_features * (
                    1 - self.title_trans_fusion_param)
        text_mask = title_mask + transcript_mask
        ### social text(title,reply..subreply....)###
        social_text_inputid = kwargs['social_text_inputid']  # (batch,n,512)
        social_text_mask = kwargs['social_text_mask']  # (batch,n,512)
        social_text_features = []
        for i in range(social_text_inputid.shape[0]):
            if 'bert' in self.bert_path:
                fea = self.bert(social_text_inputid[i], attention_mask=social_text_mask[i])[
                    'pooler_output']  # (batch,sequence,768)
            if 'AltClip' in self.bert_path:
                fea, _ = self.bert.get_text_features(social_text_inputid[i], attention_mask=social_text_mask[i])
            social_text_features.append(fea)
        social_text_features = torch.stack(social_text_features)

        # concat video and audio
        video_features = torch.cat((audio_features, video_features), dim=-1)
        video_features = self.temporal_encoder(video_features, mask=video_mask)

        # cross attention
        video_features, text_features = self.co_attention_tv(v=video_features, s=text_features,
                                                             v_len=video_features.shape[1],
                                                             s_len=text_features.shape[1],
                                                             mask_v=video_mask, mask_s=text_mask)
        # social graph
        adj = kwargs['adj']
        social_text_features = self.GAT(social_text_features, adj)  # (B,num,hidden_dim=512)

        # 平均池化
        aver_vi_features = torch.zeros(video_features.shape[0], video_features.shape[2]).cuda()
        aver_text_features = torch.zeros(text_features.shape[0], text_features.shape[2]).cuda()
        for i in range(video_features.shape[0]):
            frame_len = torch.count_nonzero(video_mask[i])
            text_len = torch.count_nonzero(text_mask[i])
            aver_vi_features[i] = torch.mean(video_features[i, :frame_len, :], -2)
            aver_text_features[i] = torch.mean(text_features[i, :text_len, :], -2)

        fea = torch.stack((aver_vi_features, aver_text_features, social_text_features[:, 0, :]), 1)

        fea = self.trm(fea)
        fea = torch.mean(fea, -2)
        output = self.classifier(fea)
        return output
