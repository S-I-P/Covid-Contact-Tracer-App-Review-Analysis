import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import random
import time
import argparse
import logging
from sys import stdout
#from pickle import dump

from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AdamW

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

config = {
    0: [BertForSequenceClassification,BertTokenizer,'bert-base-cased'],
    1: [RobertaForSequenceClassification, RobertaTokenizer,'roberta-base']
}

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT sentiment classifier')

    parser.add_argument('--model', type=int,
                        help='Which model to train? 0:Bert, 1:Roberta',
                        default=0)
    parser.add_argument('--train', type=str,
                        help='Train data file', default="train_data.xlsx")
    parser.add_argument('--test', type=str,
                        help='Test data file', default="test_data.xlsx")
    parser.add_argument('--load', type=str,
                        help='Load previously saved model', default=None)
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint Directory', default="checkpoints")
    parser.add_argument('--output', type=str,
                        help='Output Directory', default="saved_model")
    parser.add_argument('--batch', type=int,
                        help='Batch Size', default=4)
    parser.add_argument('--epoch', type=int,
                        help='Number of epochs', default=200)

    args = parser.parse_args()

    seed_torch(20200209)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')

    select_model = config[args.model]

    BATCH_SIZE = args.batch
    EPOCHS = args.epoch
    LEARNING_RATE = 1e-6
    ckpt_ep=20

    df = pd.read_excel(args.train)
    df['sentimentScore']=df['sentimentScore'].replace(-1, 2)

    tokenizer = select_model[1].from_pretrained(select_model[2], do_lower_case=True)

    reviews = df.preprocessedSA.values
    labels = df.sentimentScore.values

    MAX_LEN = 330#325

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(stdout)
        ]
    )

    #logging.info('Training Set:')
    #logging.info('Neutral Reviews: %d', np.count_nonzero(labels == 0))
    #logging.info('Positive Reviews: %d', np.count_nonzero(labels == 1))
    #logging.info('Negative Reviews: %d', np.count_nonzero(labels == 2))

    input_ids = []
    attention_masks = []

    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
                        str(review), 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt'
                   )         
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])


    train_inputs = torch.cat(input_ids, dim=0)
    train_masks = torch.cat(attention_masks, dim=0)
    train_labels = torch.tensor(labels)

    logging.info('Training data {} {} {}'.format(train_inputs.shape, train_masks.shape, train_labels.shape))
    logging.info('LEARNING_RATE: {} | Epoch: {} | BATCH_SIZE: {}'.format(LEARNING_RATE, EPOCHS, BATCH_SIZE))

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Train Model
    load = args.load
    if load is None:
        model = select_model[0].from_pretrained(select_model[2], num_labels=3)
        logging.info('Initializing model from base')
    else:
        logging.info('Loading model from {}'.format(load))
        model = select_model[0].from_pretrained(load)
    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    begin=time.time()
    train_loss_set = []

    for ep_ in trange(EPOCHS, desc="Epoch"):

        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(train_dataloader):
        
            batch = tuple(t.to(device) for t in batch)
          
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, \
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())    
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

        if ((ep_+1)%ckpt_ep)==0:
            try:
                ckpt_dir = args.ckpt+'/'+str(ep_+1)
                model.save_pretrained(ckpt_dir)
                #with open(ckpt_dir+'.pkl', 'wb') as f:
                #    dump(model, f)
            except Exception as exc:
                print(exc)

    end=time.time()
    logging.info('Training used {:.2f} second'.format(end-begin))
    try:
        model.save_pretrained(args.output)
    except Exception as exc:
        print(exc)

    ### Test
    logging.info('Starting Evaluation')
    test_df = pd.read_excel(args.test)
    test_df['sentimentScore']=test_df['sentimentScore'].replace(-1, 2)
    begin=time.time()

    reviews=test_df.preprocessedSA.values
    labels = test_df.sentimentScore.values

    input_ids = []
    attention_masks = []

    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
                        str(review), 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt'
                   )         
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    prediction_inputs = torch.cat(input_ids,dim=0)
    prediction_masks = torch.cat(attention_masks,dim=0)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

    model.eval()
    predictions,true_labels=[],[]

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)

    end=time.time()
    logging.info('Prediction used {:.2f} seconds'.format(end-begin))

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    logging.info("Accuracy of {} on APP Reviews is: {}".format(select_model[2], accuracy_score(flat_true_labels,flat_predictions)))
    logging.info("F1 score of {} on APP Reviews is: {}".format(select_model[2], f1_score(flat_true_labels,flat_predictions,average='weighted')))
    
    print(classification_report(flat_true_labels,flat_predictions))