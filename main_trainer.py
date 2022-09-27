import os
import json
import time
import nltk
import datasets
import argparse
import numpy as np
import pandas as pd

import torch

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration, 
    PreTrainedTokenizerFast
) 
from tqdm import tqdm

from utils import load_data_to_huggingface_dataset, set_seed
from IPython import embed
from time import strftime

'''
RUN: python main_trainer.py
'''

if __name__ == '__main__':
    os.environ['WANDB_DISABLED'] = 'true'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed')
    parser.add_argument('--model_backbone', default='kobart_base', type=str)
    parser.add_argument('--model_pretrained', default='gogamza/kobart-summarization', type=str, \
        help=['gogamza/kobart-base-v1', 'gogamza/kobart-base-v2', 'gogamza/kobart-summarization',\
            'hyunwoongko/asian-bart-ecjk'])
    parser.add_argument('--data_dir', default='data', type=str, help='data directory')
    parser.add_argument('--output_dir', default='checkpoint/', type=str, help='Checkpoint directory/')
    parser.add_argument('--result_dir', default='results/', type=str, help='Result directory/')

    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--wr', default=0.0, type=float, help='Warm-up ratio')
    parser.add_argument('--wd', default=0.01, type=float, help='Weight decay coefficient')
    parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size [8, 16, 32]')
    parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size [4, 8, 16, 32]')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--encoder_max_length', default=512, type=int, help='Max sequence length for encoding')
    parser.add_argument('--decoder_max_length', default=64, type=int, help='Max sequence length for decoding')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help="Label smoothing factor")

    p_args = parser.parse_args()

    start = time.time()

    # Set seed for reproducibility
    set_seed(p_args.seed)

    if not os.path.exists(p_args.result_dir):
        os.makedirs(p_args.result_dir)

    '''
    0. Set log dir
    '''
    log_dir = os.path.join(p_args.result_dir, p_args.model_backbone)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dirs = os.listdir(log_dir)

    if len(log_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split('_')[0]) for d in log_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = '%d_%s' % (idx, strftime('%Y%m%d-%H%M'))
    full_log_dir = os.path.join(log_dir, cur_log_dir)

    if not os.path.exists(full_log_dir):
        os.mkdir(full_log_dir)

    p_args.output_dir = os.path.join(full_log_dir, p_args.output_dir)

    final_result = {}
    final_result.update(p_args._get_kwargs())

    '''
    1. Load model and tokenizer
    '''
    # Download model and tokenizer
    if p_args.model_pretrained == 'kykim/bertshared-kor-base':
        from transformers import BertTokenizerFast, EncoderDecoderModel
        tokenizer = BertTokenizerFast.from_pretrained("kykim/bertshared-kor-base", model_max_length=512)
        model = EncoderDecoderModel.from_pretrained("kykim/bertshared-kor-base")
        
        model.config.min_length = None
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
    
    elif p_args.model_pretrained == 'hyunwoongko/asian-bart-ecjk':
        '''
        asian-bart library 설치가 필요합니다.
        pip install asian-bart
        '''
        from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
        tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        model = AsianBartForConditionalGeneration.from_pretrained("hyunwoongko/asian-bart-ecjk")
    
    elif p_args.model_pretrained == 'paust/pko-t5-base':
        from transformers import T5TokenizerFast, T5ForConditionalGeneration
        tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
        model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-base')
    
    elif p_args.model_pretrained == 'facebook/mbart-large-50':
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ko_KR", tgt_lang="ko_KR")

    elif p_args.model_pretrained in ['gogamza/kobart-base-v1', 'cosmoquester/bart-ko-mini', 'gogamza/kobart-summarization', 'gogamza/kobart-base-v2']:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(p_args.model_pretrained)
        # Default pre-trained model is from https://github.com/seujung/KoBART-summarization 
        model = BartForConditionalGeneration.from_pretrained(p_args.model_pretrained)
    
    else:
        print(f"Model {p_args.model_pretrained} is not supported")
        exit()

    # Set model parameters or use the default
    print(model.config)

    '''
    2. Data
    '''
    # Load dataset from local directory ('train.json', 'test.json' should be in p_args.data_dir)
    train_data, eval_data, test_data = load_data_to_huggingface_dataset(p_args.data_dir, p_args.seed, tokenizer, stratify=False, test_size=0.2)
    print(f'> Number of train data: {len(train_data)}, eval data: {len(eval_data)}, test data: {len(test_data)}')
    
    # Preprocess and tokenize data
    def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
        source, target = batch["context"], batch["summary"]
        source_tokenized = tokenizer(
            source, padding="max_length", truncation=True, max_length=max_source_length
        )
        target_tokenized = tokenizer(
            target, padding="max_length", truncation=True, max_length=max_target_length
        )

        batch = {k: v for k, v in source_tokenized.items()}
        # Ignore padding in the loss
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in l]
            for l in target_tokenized["input_ids"]
        ]
        return batch

    train_data = train_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, p_args.encoder_max_length, p_args.decoder_max_length
        ),
        batched=True,
        remove_columns=train_data.column_names,
    )

    validation_data = eval_data.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, p_args.encoder_max_length, p_args.decoder_max_length
        ),
        batched=True,
        remove_columns=eval_data.column_names,
    )



    '''
    2. Training
    '''

    # Load metrics (ROUGE-1, ROUGE-2, ROUGE-L)

    # Borrowed from https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_summarization.py

    nltk.download("punkt", quiet=True)
    metric = datasets.load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Training arguments
    # Details; https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        report_to=None,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
        output_dir=p_args.output_dir,
        num_train_epochs=p_args.num_train_epochs,  
        per_device_train_batch_size=p_args.train_batch_size,  
        per_device_eval_batch_size=p_args.test_batch_size,
        learning_rate=p_args.lr,
        weight_decay=p_args.wd,
        label_smoothing_factor=p_args.label_smoothing,
        logging_dir="logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        # metric_for_best_model="eval_rouge1", 
        # greater_is_better=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Details; https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    # Evaluate before fine-tuning
    trainer.evaluate()
    final_result.update({'before_fine_tuning': trainer.state.log_history})

    # Train the model
    trainer.train()

    '''
    3. Evaluation
    '''
    # Evaluate after fine-tuning
    trainer.evaluate()


    log_history = trainer.state.log_history

    elapsed_time = (time.time() - start) / 60 # Min.

    with open(os.path.join(full_log_dir, f'model_{p_args.model_backbone}_lr_{p_args.lr}.json'), 'w') as f:
        final_result.update(
            {
                'time': elapsed_time,
                'train_results': log_history,
                'best_results': log_history[-1],
            }
        )
        json.dump(final_result, f, indent=2)

