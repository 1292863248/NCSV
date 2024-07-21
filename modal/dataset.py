import os.path
from AltClip import AltCLIP
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NCSVDataset(Dataset):
    def __init__(self,data_path='../data',json_path='full_data.json',bert_path='../bert-base-chinese',max_length=512,use_sentiment=False):
        self.data=pd.read_json(os.path.join(data_path,json_path),orient='records',lines=True)
        if 'bert' in bert_path:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        if 'Clip' in bert_path:
            process = getattr(AltCLIP, "AltCLIPProcess").from_pretrained(os.path.join(bert_path,'checkpoints/altclip-xlmr-l'))
            self.tokenizer = process.tokenizer
        self.max_length=max_length
        self.audio_path=os.path.join(data_path,'audio_features')
        self.video_path=os.path.join(data_path,'video_features','altclip_features')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item=self.data.iloc[idx]
        vid=item['video_id']
        label=torch.tensor(item['label'])
        #标题
        title_tokens=self.tokenizer(item['title'],max_length=self.max_length, padding='max_length', truncation=True)
        title_inputid=torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])
        #转录
        if item['transcript']==[]:
            transcript_inputid=torch.LongTensor([])
            transcript_mask=torch.LongTensor([])
        else:
            transcript_tokens = self.tokenizer(item['transcript'][0], max_length=self.max_length, padding='max_length',
                                               truncation=True)
            transcript_inputid = torch.LongTensor(transcript_tokens['input_ids'])
            transcript_mask = torch.LongTensor(transcript_tokens['attention_mask'])
        #回复&子回复
        reply_count=torch.tensor(len(item['reply']))
        reply_inputid = []
        reply_mask = []
        for reply in item['reply']:
            reply_tokens=self.tokenizer(reply,max_length=self.max_length,padding='max_length',truncation=True)
            reply_inputid.append(reply_tokens['input_ids'])
            reply_mask.append((reply_tokens['attention_mask']))
        reply_inputid=torch.LongTensor(reply_inputid)
        reply_mask=torch.LongTensor(reply_mask)
        adj=item['adj']

        #视频，每1s提取一个特征
        video_features = torch.load(f'{self.video_path}/{vid}.pt', map_location='cpu')
        videoframe_count=torch.tensor(video_features.shape[0])

        audio_features = torch.load(f'{self.audio_path}/{vid}.pt', map_location='cpu')
        frame_num = video_features.shape[0]
        audio_features = audio_features[:frame_num, :]


        return {
            'label': label,
            # 标题（max_length)
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            # 转录(max_length)
            'transcript_inputid': transcript_inputid,
            'transcript_mask': transcript_mask,
            # 社交上下文(sentence_num,maxlength)
            'social_text_inputid': torch.cat((title_inputid.unsqueeze(0), reply_inputid), dim=0),
            'social_text_mask': torch.cat((title_mask.unsqueeze(0), reply_mask), dim=0),
            'social_text_count': reply_count + 1,
            'adj': adj,
            # 音频(frame_num,128)
            'audio_features': audio_features,
            # 视频(frame_num,768)
            'video_features': video_features,
            'videoframe_count': videoframe_count,
        }

def pad_frame_sequence(max_len,lst):
    max_len=min(max_len,64)#如果一个batch中有视频长度超过64，那么最大长度就取64，如果都小于64，那么就取视频长度的最大值
    result=[]
    attention_mask=[]
    for video in lst: #video(frame_num,768) or audio (frame_num,128)
        ori_len=video.shape[0]
        if ori_len>=max_len:
            gap=ori_len//max_len
            video=video[::gap][:max_len]
            mask = torch.ones(max_len)
        else:
            video=torch.cat((video,torch.zeros([max_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = torch.cat((torch.ones(ori_len), torch.zeros(max_len - ori_len)), dim=0)
        result.append(video)
        attention_mask.append(mask)
    return torch.stack(result),torch.stack(attention_mask)#(batch,frame_num,768), (batch,frame_num)

def pad_text_sequence(max_len,lst,max_length=512):#对inputid和attentionmask进行padding

    result=[]
    for text in lst:#social context (sentence_num,max_length)
        ori_len = text.shape[0]
        if ori_len<max_len:
            pad=torch.zeros(max_len-ori_len,max_length).int()
            text=torch.cat((text,pad),dim=0)
        result.append(text)
    return torch.stack(result)#(batch,sentence_num,max_length)

def collect_fn(batch):
    label=[]
    #音视频
    video_features=[]
    videoframe_count=[]
    audio_features=[]
    #标题
    title_inputid = []
    title_mask = []
    #转录
    transcript_inputid=[]
    transcript_mask=[]
    #社交上下文
    social_text_inputid=[]
    social_text_mask=[]
    social_text_count=[]
    adj=[]
    #情感
    #sentiment_feature=[]
    timesformer_features=[]
    AST_features=[]
    for data in batch:
        label.append(torch.tensor(data['label']))#(1)
        video_features.append(data['video_features'])#(frame_num,768)
        videoframe_count.append(data['videoframe_count'])#(1)
        audio_features.append(data['audio_features'])#(frame_num,128)
        title_inputid.append(data['title_inputid'])#(max_length)
        title_mask.append(data['title_mask'])#(max_length)
        transcript_inputid.append(data['transcript_inputid'])  # (max_length)
        transcript_mask.append(data['transcript_mask'])  # (max_length)
        social_text_inputid.append(data['social_text_inputid'])#(sentencenum, maxlength)
        social_text_mask.append(data['social_text_mask'])#(sentencenum, maxlength)
        social_text_count.append(data['social_text_count'])
        adj.append(data['adj'])#list (adj_num, 2)
    video_features,video_mask=pad_frame_sequence(max(videoframe_count),video_features)
    audio_features,audio_mask=pad_frame_sequence(max(videoframe_count),audio_features)
    social_text_inputid=pad_text_sequence(max(social_text_count),social_text_inputid,512)
    social_text_mask=pad_text_sequence(max(social_text_count),social_text_mask,512)
    #adj邻接矩阵构建
    adj_matrix=torch.zeros([len(batch),max(social_text_count),max(social_text_count)])#(batch,text_num,text_num)
    for i in range(len(batch)):
        edges = adj[i]#batch中第i个样本的边集
        for j in range(len(edges)):#第j条边
            adj_matrix[i,edges[j][0],edges[j][1]]=1
            adj_matrix[i, edges[j][1],edges[j][0]] = 1


    #sentiment_feature=pad_frame_sequence(max(social_text_count),sentiment_feature)

    return{
        'label': torch.stack(label),#(batch,1)
        'title_inputid': torch.stack(title_inputid),#(batch,maxlength)
        'title_mask': torch.stack(title_mask),#(batch,maxlength)
        'transcript_inputid': torch.stack(transcript_inputid),#(batch,maxlength)
        'transcript_mask': torch.stack(transcript_mask),#(batch,maxlength)
        'social_text_inputid':social_text_inputid,#(batch,sentence_num,maxlength)
        'social_text_mask':social_text_mask,#(batch,sentence_num,maxlength)
        'adj': adj_matrix,  # (batch,node_num,node_num)
        'video_features':video_features,#(batch,frame_num,768)
        'video_mask':video_mask,#(batch,frame_num)
        'audio_features':audio_features,#(batch,frame_num,128),
        'audio_mask':audio_mask,#(batch,frame_num)
    }
