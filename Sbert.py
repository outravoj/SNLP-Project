import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


MODEL_NAME = "bert-base-cased"
MAX_LEN = 50 # based on exploratory data analysis


class Sbert(nn.Module):

  def __init__(self,n_classes, evaluate = False):
    super(Sbert, self).__init__()
    self.bert = BertModel.from_pretrained(MODEL_NAME) ##our Siamese network
    self.drop = nn.Dropout(p=0.2)
    self.out = nn.Linear(self.bert.config.hidden_size*3, n_classes) # sentence1 embedding, sentence2 embedding, their difference



  def forward(self, input_ids, attention_masks):
    sentences1 = input_ids[:,0,:]
    sentences2 = input_ids[:,1,:]
    attention_mask_1 = attention_masks[:,0,:]
    attention_mask_2 = attention_masks[:,1,:]
    model_output_sentences1 = self.bert(
      input_ids=sentences1,
      attention_mask = attention_mask_1,
      output_hidden_states=True
    )
    model_output_sentences2 = self.bert(
      input_ids=sentences2,
      attention_mask = attention_mask_2,
      output_hidden_states=True
    )
    sentence1_emb = torch.mean(model_output_sentences1.hidden_states[12], dim = 1)
    sentence2_emb = torch.mean(model_output_sentences2.hidden_states[12], dim = 1)
    diff = abs(sentence1_emb-sentence2_emb)
    concats = torch.cat([sentence1_emb, sentence2_emb], dim = 1)
    concats = torch.cat([concats, diff], dim = 1)
    return self.out(concats)

  def embed(self, input_ids):
    model_output_sentences = self.bert(
      input_ids=input_ids,
      output_hidden_states=True
    )
    return torch.mean(model_output_sentences.hidden_states[12], dim = 1)

def pass_through_sbert(sentence: str, tokenizer: BertTokenizer, sbert: Sbert, device):
  encoding = tokenizer(sentence, max_length = MAX_LEN, padding = "max_length", truncation = True)
  input_ids = torch.tensor(encoding["input_ids"], dtype = torch.int)
  input_ids = input_ids.to(device)
  return sbert.embed(input_ids[None,:])
