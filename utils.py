import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import os
import json
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pandas as pd

import datasets
from IPython import embed
from datasets import Dataset
from transformers.utils import logging

logger = logging.get_logger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def load_data_to_huggingface_dataset(data_dir, seed, tokenizer, stratify=True, test_size=0.2):
    print(f"> Loading data from {data_dir}")
    TRAIN_SOURCE = os.path.join(data_dir, "train.json")
    TEST_SOURCE = os.path.join(data_dir, "test.json")

    with open(TRAIN_SOURCE) as f:
        TRAIN_DATA = json.loads(f.read())
        
    with open(TEST_SOURCE) as f:
        TEST_DATA = json.loads(f.read())

    train = pd.DataFrame(columns=['uid', 'title', 'region', 'context', 'summary'])
    uid = 1000
    for data in TRAIN_DATA:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            train.loc[uid, 'uid'] = uid
            train.loc[uid, 'title'] = data['title']
            train.loc[uid, 'region'] = data['region']
            train.loc[uid, 'context'] = context[:-1]
            train.loc[uid, 'summary'] = data['label'][agenda]['summary']
            uid += 1

    test = pd.DataFrame(columns=['uid', 'title', 'region', 'context'])
    uid = 2000
    for data in TEST_DATA:
        for agenda in data['context'].keys():
            context = ''
            for line in data['context'][agenda]:
                context += data['context'][agenda][line]
                context += ' '
            test.loc[uid, 'uid'] = uid
            test.loc[uid, 'title'] = data['title']
            test.loc[uid, 'region'] = data['region']
            test.loc[uid, 'context'] = context[:-1]
            uid += 1

    if stratify:
        # https://dacon.io/competitions/official/235813/codeshare/3719?page=1&dtype=recent ??????
        # context ?????? ?????? 
        def token_len(text):
            return len(tokenizer.tokenize(text))

        # context??? ????????? ?????? ??????, ?????? ?????? ??????, ?????? ??????, ????????? ???????????? ??????
        def type_classifier(context):
            if '??????' in context[:1000]:
                return '?????? ??????'
            elif (len(context.split('????????? ???')) > 2 and len(tokenizer.tokenize(context)) > 1024 and '??????' not in context[:200]):#and '??????' not in summary[-3:]:
                return '?????? ?????? ??????' 
            elif '????????????' in context[:200] and len(context.split('????????? ???')) > 1:
                return '????????????'
            elif '??????' in context[:200]:
                return '?????? ??????'
            elif '??????' in context[:100]:
                return '?????? ??????'
            elif '??????' in context[:200]:
                return '?????? ??????'
            else:
                return '??????' 

        # train,test??? ?????? ?????? ????????? ????????? ?????? ?????? ??????
        train['con_token_len'] = train['context'].apply(token_len)
        train['con_type'] = train['context'].apply(type_classifier)

        test['con_token_len'] = test['context'].apply(token_len)
        test['con_type'] = test['context'].apply(type_classifier)

        # convert to Huggingface dataset
        train = train[['context', 'summary', 'con_type']]
        test = test[['context', 'con_type']]

        train_dataset = Dataset(pa.Table.from_pandas(train))
        test_dataset = Dataset(pa.Table.from_pandas(test))

        train_dataset = train_dataset.class_encode_column("con_type")
        test_dataset = test_dataset.class_encode_column("con_type")
        
        try:
            train_dataset = train_dataset.remove_columns('__index_level_0__')
            test_dataset = test_dataset.remove_columns('__index_level_0__')
        except:
            pass

        train_data, eval_data = train_dataset.train_test_split(test_size=test_size, shuffle=True, seed=seed, stratify_by_column='con_type').values()

    else:
        # convert to Huggingface dataset
        train = train[['context', 'summary']]
        test = test[['context']]

        train_dataset = Dataset(pa.Table.from_pandas(train))
        test_dataset = Dataset(pa.Table.from_pandas(test))

        try:
            train_dataset = train_dataset.remove_columns('__index_level_0__')
            test_dataset = test_dataset.remove_columns('__index_level_0__')
        except:
            pass

        train_data, eval_data = train_dataset.train_test_split(test_size=test_size, shuffle=True, seed=seed).values()

    del TRAIN_DATA, TEST_DATA
    return train_data, eval_data, test_dataset
