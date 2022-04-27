import torch
from torch import nn
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


MODEL_NAME = "google/electra-small-discriminator"
MAX_LEN = 50 # based on exploratory data analysis


class SELECTRA(nn.Module):

  def __init__(self,n_classes, evaluate = False):
    super(SELECTRA, self).__init__()
    self.electra = ElectraModel.from_pretrained(MODEL_NAME) ##our Siamese network
    self.drop = nn.Dropout(p=0.2)
    self.out = nn.Linear(self.electra.config.hidden_size*3, n_classes) # sentence1 embedding, sentence2 embedding, their difference



  def forward(self, input_ids, attention_masks):
    sentences1 = input_ids[:,0,:]
    sentences2 = input_ids[:,1,:]
    attention_mask_1 = attention_masks[:,0,:]
    attention_mask_2 = attention_masks[:,1,:]
    model_output_sentences1 = self.electra(
      input_ids=sentences1,
      attention_mask = attention_mask_1,
      output_hidden_states=True
    )
    model_output_sentences2 = self.electra(
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

  def embed(self, input_ids, attention_mask):
    model_output_sentences = self.electra(
      input_ids=input_ids,
      attention_mask = attention_mask,
      output_hidden_states=True
    )
    return torch.mean(model_output_sentences.hidden_states[12], dim = 1)

def pass_through_selectra(sentence: str, tokenizer: ElectraTokenizer, selectra: SELECTRA, device):
  encoding = tokenizer(sentence, max_length = MAX_LEN, padding = "max_length", truncation = True, return_attention_mask = True)
  input_ids = torch.tensor(encoding["input_ids"], dtype = torch.int)
  attention_mask = torch.tensor(encoding["attention_mask"], dtype = torch.int)
  attention_mask = attention_mask.to(device)
  input_ids = input_ids.to(device)
  return selectra.embed(input_ids[None,:], attention_mask[None,:])
