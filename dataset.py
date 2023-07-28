import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset , tokenizer_src , tokenizer_tgt , src_lang , tgt_lang ,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) :
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform dataset into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids


        # Add sos eos and padding to each sentence 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2 for sos and eos
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 for sos

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too short')
        
        # Add sos and eos to the beginning and end of each sentence
        encoder_input = torch.cat(
            [
                self.sos_token ,
                torch.tensor(enc_input_tokens , dtype=torch.int64) ,
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens , dtype=torch.int64)
            ],
             dim=0
        )

        # Add sos only to the beginning of each sentence in decoder 
        decoder_input = torch.cat(
            [
                self.sos_token ,
                torch.tensor(dec_input_tokens , dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
            ],
            dim=0
        )

        # Add eos only for label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
            ]
        )

        # check if the size of the tensors is the same to seq_len 
        assert len(encoder_input) == self.seq_len
        assert len(decoder_input) == self.seq_len
        assert label.size(0) == self.seq_len


        return {
            'encoder_input' : encoder_input ,
            'decoder_input' : decoder_input ,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1 ,1 seq_len)
            'decoder_mask': (decoder_input !=self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label' : label,
            'src_text' : src_text,
            'tgt_text' : tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size, size), diagonal=1).type(torch.int)
    return mask == 0