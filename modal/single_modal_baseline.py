import torch
import torch.nn as nn
from transformers import BertModel

from AltClip import AltCLIP
from modal.coattention import co_attention


class bert_base_chinese(nn.Module):
    def __init__(self, bert_path, fea_dim=512, dropout=0.1,text='title'):
        super(bert_base_chinese, self).__init__()
        self.text_dim = 768
        self.text=text
        self.attention = nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, batch_first=True)

        self.bert = BertModel.from_pretrained(bert_path).requires_grad_(False)

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim))
        self.classifier=nn.Sequential(torch.nn.ReLU(), nn.Linear(fea_dim, 2))

    def forward(self, **kwargs):

        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']
        transcript_inputid = kwargs['transcript_inputid']
        transcript_mask = kwargs['transcript_mask']
        social_text_inputid=kwargs['social_text_inputid']
        social_text_mask=kwargs['social_text_mask']
        reply_feature=[]
        if self.text=='title':
            fea_text = self.bert(title_inputid, attention_mask=title_mask)[1]
            fea_text = self.linear_text(fea_text)
            output = self.classifier(fea_text)
        elif self.text=='transcript':
            fea_text = self.bert(transcript_inputid, attention_mask=transcript_mask)[1]
            fea_text = self.linear_text(fea_text)
            output = self.classifier(fea_text)
        else:
            for i in range(social_text_inputid.shape[0]):
                fea_text = self.bert(social_text_inputid[i], attention_mask=social_text_mask[i])[1]
                reply_feature.append(fea_text)
            reply_feature = torch.stack(reply_feature)
            reply_feature = self.linear_text(reply_feature)
            reply_feature = self.attention(reply_feature)
            fea_text = torch.mean(reply_feature, -2)
            output = self.classifier(fea_text)

        return output

class clip_image_encoder(torch.nn.Module):
    def __init__(self,fea_dim=512):
        super(clip_image_encoder,self).__init__()
        self.image_dim=768
        self.atten=nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, batch_first=True)
        self.linear_img = nn.Sequential(torch.nn.Linear(self.image_dim, fea_dim),torch.nn.ReLU())
        self.classifier = nn.Linear(fea_dim,2)

    def forward(self, **kwargs):
        fea_img = self.linear_img(kwargs['video_features'])
        fea_img = self.atten(fea_img)
        video_mask = kwargs['video_mask']
        aver_vi_features = torch.zeros(fea_img.shape[0], fea_img.shape[2]).cuda()
        for i in range(fea_img.shape[0]):
            frame_len=torch.count_nonzero(video_mask[i])
            aver_vi_features[i]=torch.mean(fea_img[i,:frame_len,:],-2)

        output = self.classifier(aver_vi_features)
        return output

class clip_text_encoder(nn.Module):
    def __init__(self, bert_path, fea_dim=512, dropout=0.1,text='title'):
        super(clip_text_encoder, self).__init__()
        self.text_dim = 768
        self.text = text

        self.bert = getattr(AltCLIP, "AltCLIP").from_pretrain(
            download_path=f"{bert_path}/checkpoints",
            model_name="altclip-xlmr-l",
            only_download_config=False,
            device="gpu", ).requires_grad_(False)
        self.attention = nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, batch_first=True)
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim))
        self.classifier = nn.Sequential(torch.nn.ReLU(),nn.Linear(fea_dim, 2))

    def forward(self, **kwargs):
        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']
        transcript_inputid = kwargs['transcript_inputid']
        transcript_mask = kwargs['transcript_mask']
        social_text_inputid = kwargs['social_text_inputid']
        social_text_mask = kwargs['social_text_mask']
        reply_feature = []
        if self.text == 'title':
            fea_text,_ = self.bert.get_text_features(title_inputid, attention_mask=title_mask)
            fea_text = self.linear_text(fea_text)
            output = self.classifier(fea_text)
        elif self.text=='transcript':
            fea_text,_ = self.bert.get_text_features(transcript_inputid, attention_mask=transcript_mask)
            fea_text = self.linear_text(fea_text)
            output = self.classifier(fea_text)
        else:
            for i in range(social_text_inputid.shape[0]):
                fea_text,_ = self.bert.get_text_features(social_text_inputid[i], attention_mask=social_text_mask[i])
                reply_feature.append(fea_text)
            reply_feature = torch.stack(reply_feature)
            reply_feature = self.linear_text(reply_feature)[:,1:,:]
            reply_feature = self.attention(reply_feature)
            fea_text = torch.mean(reply_feature, -2)
            output = self.classifier(fea_text)
        return output


class Vggish(torch.nn.Module):
    def __init__(self, fea_dim=128):
        super(Vggish, self).__init__()
        self.audio_dim = 128
        self.atten = nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, batch_first=True)
        self.linear_audio = torch.nn.Linear(self.audio_dim, fea_dim)
        self.classifier = nn.Sequential(torch.nn.ReLU(), nn.Linear(fea_dim, 2))

    def forward(self, **kwargs):
        fea_audio = kwargs['audio_features']
        fea_audio = self.linear_audio(fea_audio)
        fea_audio = self.atten(fea_audio)
        fea_audio = torch.mean(fea_audio, -2)
        output = self.classifier(fea_audio)
        return output

class AST(torch.nn.Module):
    def __init__(self,fea_dim=512):
        super(AST,self).__init__()
        self.audio_dim = 768
        self.linear_audio = torch.nn.Linear(self.audio_dim, fea_dim)
        self.classifier = nn.Sequential(torch.nn.ReLU(), nn.Linear(fea_dim, 2))

    def forward(self, **kwargs):
        fea_audio = kwargs['AST_features']
        fea_audio = self.linear_audio(fea_audio)
        output = self.classifier(fea_audio)
        return output




class TimeSFormer(torch.nn.Module):
    def __init__(self,fea_dim=512):
        super(TimeSFormer,self).__init__()
        self.video_dim = 768
        self.linear_video = torch.nn.Linear(self.video_dim, fea_dim)
        self.atten = nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, batch_first=True)
        self.classifier = nn.Sequential(torch.nn.ReLU(), nn.Linear(fea_dim, 2))

    def forward(self, **kwargs):
        fea_video = kwargs['timesformer_features']
        fea_video = self.linear_video(fea_video)
        output = self.classifier(fea_video)
        return output



class SVFENDModel(torch.nn.Module):
    def __init__(self,  fea_dim=512, dropout=0.1):
        super(SVFENDModel, self).__init__()

        self.text_dim = 1024
        self.comment_dim = 768
        self.img_dim = 768
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4
        self.audio_dim = 128

        self.dropout = dropout


        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                            pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(fea_dim, 2)


    def forward(self, **kwargs):



        fea_text = kwargs['text_feature']
        fea_text = self.linear_text(fea_text)
        ### Audio Frames ###
        audioframes = kwargs['audio_features']  # (batch,36,12288)
        # audioframes_masks = kwargs['audioframes_masks']
        fea_audio = self.linear_audio(audioframes)
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1],
                                                   s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        ### Image Frames ###
        video_feature= kwargs['video_features']  # (batch,30,768)
        fea_img = self.linear_img(video_feature)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)

        fea_comments=kwargs['social_features'][:,1:,:]
        fea_comments = self.linear_comment(fea_comments)  # (batch,fea_dim)
        fea_comments=torch.mean(fea_comments,-2)


        fea_text = fea_text.unsqueeze(1)
        fea_comments = fea_comments.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)


        fea = torch.cat((fea_text, fea_audio, fea_img, fea_comments), 1)  # (bs, 6, 128)
        fea = self.trm(fea)
        fea = torch.mean(fea, -2)

        output = self.classifier(fea)


        return output