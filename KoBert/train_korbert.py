import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from imageio import imread
from PIL import Image
import matplotlib.image as mpimg
from IPython import display
from transformers import BertModel
#from kobert_tokenizer import KoBERTTokenizer
#from kobert.utils import get_tokenizer
#from kobert.pytorch_kobert import get_pytorch_kobert_model
from tokenization_kobert import KoBertTokenizer
import gluonnlp as nlp
import os

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device("cuda:0")

#model, vocab = get_pytorch_kobert_model()
model = BertModel.from_pretrained("monologg/kobert")

tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
#tokenizer = get_tokenizer()
#tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
#tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

#####################################
# Encoder RASNET CNN - pretrained
#####################################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out
    
####################
# Attention Decoder
####################
class Decoder(nn.Module):

    def __init__(self, vocab_size, use_glove, use_bert):
        super(Decoder, self).__init__()
        self.encoder_dim = 2048
        self.attention_dim = 512
        self.use_bert = use_bert
        if use_glove:
            self.embed_dim = 300
        elif use_bert:
            self.embed_dim = 768
        else:
            self.embed_dim = 512
            
        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.5
        
        # soft attention
        self.enc_att = nn.Linear(2048, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # init variables
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True
            

    def forward(self, encoder_out, encoded_captions, caption_lengths):    
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        dec_len = [x-1 for x in caption_lengths]
        max_dec_len = max(dec_len)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        if not self.use_bert:
            embeddings = self.embedding(encoded_captions)
        elif self.use_bert:
            
            embeddings = []
            for cap_idx in  encoded_captions:
                
                # padd caption to correct size
                while len(cap_idx) < max_dec_len:
                    cap_idx.append(PAD)
                    
                cap = ' '.join([vocab.idx2word[word_idx.item()] for word_idx in cap_idx])
                cap = u'[CLS] '+cap
                
                tokenized_cap = tokenizer.tokenize(cap)                
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                
                tokens_tensor = torch.tensor([indexed_tokens])
                
                with torch.no_grad():
                    #encoded_layers, just = model(tokens_tensor)
                    model(tokens_tensor)
                    bert_embedding = model(tokens_tensor)[0][0].squeeze(0)
                
                split_cap = cap.split()
                tokens_embedding = []
                j = 0

                for full_token in split_cap:
                    curr_token = ''
                    x = 0
                    for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
                        token = tokenized_cap[i+j]
                        piece_embedding = bert_embedding[i+j]
                        
                        # full token
                        if token == full_token and curr_token == '' :
                            tokens_embedding.append(piece_embedding)
                            j += 1
                            break
                        else: # partial token
                            x += 1
                            
                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token.replace('#', '')
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token.replace('#', '')
                                
                                if curr_token == full_token: # end of partial
                                    j += x
                                    break                            
                
                              
                cap_embedding = torch.stack(tokens_embedding)

                embeddings.append(cap_embedding)
            
            embeddings = torch.stack(embeddings)

        # init hidden state
        avg_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(avg_enc_out)
        c = self.c_lin(avg_enc_out)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels)

        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len ])
            
            # soft-attention
            enc_att = self.enc_att(encoder_out[:batch_size_t])
            dec_att = self.dec_att(h[:batch_size_t])
            att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)
        
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            batch_embeds = embeddings[:batch_size_t, t, :]  
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1)
            
            h, c = self.decode_step(cat_val.float(),(h[:batch_size_t].float(), c[:batch_size_t].float()))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        # preds, sorted capts, dec lens, attention wieghts
        return predictions, encoded_captions, dec_len, alphas
    
# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

PAD = 0
START = 1
END = 2
UNK = 3

grad_clip = 5.
num_epochs = 4
batch_size = 16
decoder_lr = 0.0004

# Load vocabulary wrapper
with open('/root/volume/coco2014/kor_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('val', vocab, batch_size)

criterion = nn.CrossEntropyLoss()


decoder = Decoder(vocab_size=len(vocab),use_glove=False, use_bert=True) # true = use glove
decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

encoder = Encoder()

def train():
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets)
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))
            print('Batch '+str(i)+'/'+str(num_batches)+' loss:'+str(losses.avg)+'\n')

            
            # save model each 100 batches
            if i%60==0 and i!=0:
                print('Batch '+str(i)+'/'+str(num_batches)+' loss:'+str(losses.avg)+'\n')
                
                
                save_decoder_path = os.path.join('/root/volume/Capstone/OURS/checkpoints/', "decoder"+str(epoch)+"_batch16.pt")
                save_encoder_path = os.path.join('/root/volume/Capstone/OURS/checkpoints/', "encoder"+str(epoch)+"_batch16.pt")
                 # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                    }, save_decoder_path)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                    }, save_encoder_path)

                print('model saved')
        
        print('Epoch Loss:'+str(losses.avg))

    print("Completed training...")  

train()