import os
import torch 
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split
from datasets import load_dataset
from tokenizers import Tokenizer
import torchmetrics
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset , causal_mask
from pathlib import Path 
from model import build_transformer
from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
def get_all_sentences(dataset , lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config , dataset , lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # replace the unknow token with UNK
        tokenizer.pre_tokenizer = Whitespace() 
        trainer = WordLevelTrainer(special_tokens = ["[PAD]", "[UNK]", "[EOS]", "[SOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang) , trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}",split='train')

    ## building a tokenizer 
    tokenizer_src = get_or_build_tokenizer(config , ds_raw , config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config['lang_tgt'])

    # spliting the dataset into 90% training and 10% validation
    train_dataset_size = int(len(ds_raw) * 0.9)
    val_dataset_size = len(ds_raw) - train_dataset_size

    train_dataset_raw , val_dataset_raw = random_split(ds_raw , [train_dataset_size , val_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # find the max length of the sentences in source and target sentence

    max_len_src = 0 
    max_len_tgt = 0 

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src , len(src_ids))
        max_len_tgt = max(max_len_tgt , len(tgt_ids))


    # printing the max_length of source and target sentence
    print(f"max length of source sentence : {max_len_src}")
    print(f"max length of target sentence : {max_len_tgt}")

    data_loader = DataLoader(train_dataset , batch_size = config['batch_size'] , shuffle = True)
    val_loader = DataLoader(val_dataset , batch_size = 1 , shuffle = True)

    return data_loader , val_loader , tokenizer_src , tokenizer_tgt



def get_model(config ,src_vocab_len , tgt_vocab_len):
    model = build_transformer(src_vocab_len , tgt_vocab_len , config['seq_len'] , config['seq_len'],d_model=config['d_model'])
    return model

def train_model(config):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device : {device}")


    # Make sure the weight foler exist 
    Path(config['model_folder']).mkdir(parents = True , exist_ok = True)
    
    train_data_loader , val_data_loader , tokenizer_src , tokenizer_tgt = get_dataset(config)

    model = get_model(config , tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    ## Tensorboard 
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters() , lr = config['learning_rate'],eps=1e-9) 


    # if the user specify a model to preload before training 
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config , config=['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_tgt.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_data_loader,desc = f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensors through the encoder , decoder and projecton layer 
            encoder_output = model.encode(encoder_input , encoder_mask)
            decoder_output = model.decode(encoder_output , encoder_mask , decoder_input , decoder_mask)
            proj_output = model.projection(decoder_output)


            label = batch['label'].to(device)
            
            ## loss calculation 
            loss = loss_fn(proj_output.view(-1 , tokenizer_tgt.get_vocab_size()) , label.view(-1))
            batch_iterator.set_postfix({"loss" : f"{loss.item():6.3f}"})


            # log the loss 
            writer.add_scalar("train_loss",loss.item() , global_step)
            writer.flush()

            # backpropation
            loss.backward()
            
            # update the weights    
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step +=1 
        



def run_validation(model ,validation_dataset ,tokenizer_src, 
                   tokenizer_tgt , max_len , device , print_msg ,global_step ,writer , num_examples =2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80


    with torch.no_grad():
        for batch in validation_dataset:
            count += 1 
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            

            # check that the batch size is 1 
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decoding(model , encoder_input, encoder_mask
                                         ,tokenizer_src,tokenizer_tgt , max_len , device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
            if writer:
                # Evaluate the character error rate
                # Compute the char error rate 
                metric = torchmetrics.CharErrorRate()
                cer = metric(predicted, expected)
                writer.add_scalar('validation cer', cer, global_step)
                writer.flush()

                # Compute the word error rate
                metric = torchmetrics.WordErrorRate()
                wer = metric(predicted, expected)
                writer.add_scalar('validation wer', wer, global_step)
                writer.flush()

                # Compute the BLEU metric
                metric = torchmetrics.BLEUScore()
                bleu = metric(predicted, expected)
                writer.add_scalar('validation BLEU', bleu, global_step)
                writer.flush()


def greedy_decoding(model , source , source_mask ,tokenizer_src,tokenizer_tgt , max_len , device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source , source_mask)
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True: 
        if decoder_input.size(1) == max_len:
            break 

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        out = model.decode(encoder_output, source_mask , decoder_input , decoder_mask)

        prob = model.project(out[:,-1])
        _, next_word = torch.max(prob , dim = 1)
        decoder_input = torch.cat(
            [decoder_input , torch.empty(1,1).fill_(next_word.item()).type_as(source).to(device)],dim=1
        )

        if next_word == eos_idx:
            break 
    return decoder_input.squeeze(0)