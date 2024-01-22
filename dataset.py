from myutils import *
# from arguments import myargs
# from pretrain import logger
import torch
import csv
import os,sys
import re
import json
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel)
from transformers.file_utils import PaddingStrategy
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
import traceback, pdb, random
from copy import deepcopy
np.set_printoptions(threshold= sys.maxsize)
import nltk
nltk.data.path.append('./nltk')
from nltk.tokenize import sent_tokenize
import difflib

type_dict = {'reddit':0, 'wikidialog':1, 'disturbwiki':2}

def print_sample(input_sen, output_sen=None, task_type='', outdir='./'):
    if outdir is not None:
        outfile=open(os.path.join(outdir,'data_sample.log'),'a')
    else:
        outfile=open(os.path.join('./','data_sample.log'),'a')

    print(f'-'*100, file=outfile)
    print(f'Task type: {task_type}', file=outfile)
    print(f'Input sentence:\n {input_sen}', file=outfile)
    if output_sen is not None:
        print(f'Output sentence:\n {output_sen}', file=outfile)
    print(f'*'*100, file=outfile)

def get_tokenized_batch(batch, pretraintype='add_responding', tokenizer=None, pp=False, task_type='response', max_source_length=1024, 
                        max_target_length=256, label_pad_token_id=0, outdir=None):
    batch_with_prompt = []
    selected = []
    typelist = []
    for i in batch:
        aa = apply_prompt(i, type=pretraintype)
        if aa:
            batch_with_prompt += aa
            selected.append(i)
            typelist.append(type_dict[i['datatype']])

    if not len(batch_with_prompt): 
        return None

    input_sen = [i['input'] for i in batch_with_prompt]
    output_sen = [i['output'] for i in batch_with_prompt]
    if pp: print_sample(input_sen[0], output_sen[0], task_type=task_type, outdir=outdir)

    encoding = tokenizer(input_sen, padding='longest',max_length=max_source_length, truncation=True, return_tensors='pt') # padding='longest'
    input_ids, attn_mask = encoding.input_ids, encoding.attention_mask
    target = tokenizer(output_sen, padding='longest', max_length=max_target_length, truncation=True, return_tensors='pt').input_ids
    target[target==tokenizer.pad_token_id] = label_pad_token_id

    nli_weights = [1 for i in batch]
    res_batch = {'input_ids':input_ids,'attention_mask':attn_mask,'labels':target,'nli_weights':nli_weights, 'batch':selected, 'typelist':typelist}
    return res_batch


# outfile = open(os.path.join(training_args.output_dir,'printed_data.log'),'a')
class Doc2DialDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_input_len=1024, tokz=None, datatype='reddit', num_data=int(1e10), outdir=None, myargs=None):
        self.tokenizer = tokz
        self.data_path = data_path
        self.max_input_len = max_input_len
        self.type = datatype
        self.num_data = num_data
        self.outdir = outdir
        self.myargs = myargs

        self.data = []

        with open(data_path, 'r') as f:
            for i in f:
                sample = json.loads(i)
                try:
                    ss = self.per_sample(sample, print_data=False)
                except Exception:
                    ss = None
                if ss is not None:
                    self.data.append(ss) 

        print(f'Finish processing data from {data_path}', flush=True)

        self.data = self.data[:self.num_data]
 
    def per_sample(self,sample, print_data=False):
        # logger.info(f'sample {sample}')
        if 'nli_score' in sample and self.myargs.nli_weighted:
            nli_weight = sample['nli_score']
        else:
            nli_weight = 1.0

        dialog_list = [] 
        response =  sample['response'].replace('\n',' ').replace('\\',' ').replace('\t',' ')

        if 'turns' in sample:
            for user in sample['turns'].values():
                for sen in user.values():
                    dialog_list.append(sen) 
            response = '<agent> ' + response
        else:
            dialog_list = sample['utterances']
            response = '<grounding> ' + response

        if len(dialog_list) == 1:
            history = ""
        else:
            history = ' '.join(dialog_list[:-1]).replace('\n',' ').replace('\\',' ').replace('\t',' ')

        utterance = dialog_list[-1].replace('\n',' ').replace('\\',' ').replace('\t',' ')
        utterance = '<last_turn> ' + utterance

        # document = ' '.join(sample['related_passage']).replace('\n',' ').replace('\\',' ').replace('\t',' ')
        document = sample['related_passage'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
        if len(document)<3:
            return None
        document = '<title>Document</title> ' + document

        context_list = []
        odd = True if len(dialog_list)%2==1 else False
        for i in range(len(dialog_list)-1,-1,-1):
            if i % 2 == 0:
                if odd:
                    context_list.append('<user> ' + dialog_list[i])
                else:
                    context_list.append('<agent> ' + dialog_list[i])
            else:
                if odd: 
                    context_list.append('<agent> ' + dialog_list[i])
                else:
                    context_list.append('<user> ' + dialog_list[i])
        context = ' '.join(context_list)     

        res_dict = {
            'context': context,
            'utterance': utterance, 
            'response': response,
            'history': history,
            'document': document,
            'datatype': self.type,
            'nli_weight': nli_weight
        }
        # print(f'sample dict {res_dict}', flush=True)
        if print_data and self.outdir:
            outfile=open(os.path.join(self.outdir,'printed_data.log'),'a')
            print(f'Sample:\n {json.dumps(sample, ensure_ascii=False)}', file=outfile)
            print(f'Returned_dictionary:\n {json.dumps(res_dict, ensure_ascii=False)}', file=outfile)

        return res_dict

    def data_process(self, data):
        printdata = True
        outdata = []
        for i in data:
            if i is not None:
                if self.myargs.nli_selected:
                    # print(i.keys(),flush=True)
                    if 'nli_score' in i:
                        nli_score = i['nli_score']
                        if nli_score < self.myargs.nli_threshold:
                            continue
                    else:
                        continue
                    sample = self.per_sample(i, print_data=printdata)
                else:
                    sample = self.per_sample(i, print_data=printdata)
                if sample is not None:
                    outdata.append(sample)
                printdata = False
        return outdata    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class MixedDataset(Doc2DialDataset):
    def __init__(self, datalist, max_input_len=1024, tokz=None):
        self.tokenizer = tokz
        self.max_input_len = max_input_len

        self.data = []
        for dataset in datalist:
            if dataset is not None:
                self.data += dataset.data
        

class PinnedBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pin_memory(self):
        for k in self.data.keys():
            if type(self.data[k])!=list:
                self.data[k] = self.data[k].pin_memory()
        return self

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()
    
    def len(self):
        return len(self.data)


# Modified by https://github.com/huggingface/transformers/blob/2e4082364e4bd001f7933d81b3f75548704f79d7/examples/flax/language-modeling/run_t5_mlm_flax.py#L212
@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel]
    pad_token_id: int
    decoder_start_token_id: int
    noise_density: float = 0.15
    mean_noise_span_length: float = 3
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pp: bool = True

    def __call__(self, batch) -> Dict[str, np.ndarray]:
        """ Make T5 MLM Batch
        Args:
            features (List[Dict[str, List]]):
                - input_ids
                - attention_mask
        Returns:
            [type]: [description]
        """
        input_sen = []
        batch11 = []
        for i in batch:
            # print('mlm',i, flush=True)
            aa = apply_prompt(i, type='add_understanding', task='understand_dialog')
            if aa:
                input_sen += aa
                batch11.append(i)
            bb = apply_prompt(i, type='add_understanding', task='understand_document')
            if bb:
                input_sen += bb
                batch11.append(i)
        # if self.pp: print_sample(input_sen[0], task_type='understanding')
        input_ids = [self.tokenizer.encode(i['input'], max_length=self.max_length, truncation=True, return_tensors='pt') for i in input_sen]
        # logger.info(f'{input_sen[0]}')
        # logger.info(f'{input_ids[0]}')
        nli_weights = [1.0 for i in range(len(input_sen))]

        # logger.info(f'length of tokenizer {len(self.tokenizer)}')
        masked_input_ids = []
        target_ids = []
        for sample_input in input_ids:
            # logger.info(f'input_ids {input_ids}')
            # input_ids = input_encoding.input_ids
            # logger.info(f'mlm input shape {sample_input.shape}')
            # expandend_input_length = sample_input.shape[0]
            # batch_size = 1
            batch_size, expandend_input_length = sample_input.shape

            mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            new_input = self.filter_input_ids(sample_input, input_ids_sentinel)
            new_label = self.filter_input_ids(sample_input, labels_sentinel)
            # logger.info(f'new input {new_input}, type is {type(new_input)}, label {new_label}')
            masked_input_ids.append(new_input[0].tolist())
            target_ids.append(new_label[0].tolist())

        input_max_length = max([len(i) for i in masked_input_ids])
        label_max_length = max([len(i) for i in target_ids])
        padded_input_ids = [i + [self.tokenizer.pad_token_id]*(input_max_length-len(i)) for i in masked_input_ids]

        labels = [i+[-100]*(label_max_length-len(i)) for i in target_ids]

        features = {
            'input_ids': torch.tensor(padded_input_ids),
            'labels': torch.tensor(labels).contiguous(),'batch': batch11}
        features['attention_mask'] = features['input_ids'] > 0
        features['nli_weights'] = nli_weights

        return features

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - 6 - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def generate_target_ids(self, input_ids, mask_prob=0.15):
        extra_tokens = [f"<extra_id_{i}>" for i in range(0, 100)]
        mask_tokens = self.tokenizer.convert_tokens_to_ids(extra_tokens)

        masked_input_ids = []
        target_ids = []
        for _input_ids in input_ids:  # let's calculate masks for denoising pretraining
            # logger.info(f'sample-->{_input_ids}')
            _input_sent_embed = deepcopy(_input_ids) 
            _target_sent_embed = []
            masked_indexes = sorted(random.sample(range(0, len(_input_sent_embed)),  # sample a word index in sentence
                                                min(int(mask_prob * len(_input_sent_embed)),  # number of tokens masked
                                                    len(mask_tokens) - 1)))  # but never more than special tokens available
            mask = [(i in masked_indexes)  # this is True or False
                    for i in range(len(_input_sent_embed))]
            i = 0
            end = len(_input_sent_embed)
            masked_spans_counter = 0
            while i < end:
                if mask[i]:
                    current_words_masked = [_input_sent_embed[i]]
                    _input_sent_embed[i] = mask_tokens[masked_spans_counter]
                    masked_spans_counter += 1
                    while i + 1 < end and mask[i + 1]:
                        current_words_masked.append(_input_sent_embed[i + 1])
                        del _input_sent_embed[i + 1]
                        del mask[i + 1]
                        end -= 1
                    _target_sent_embed.extend(current_words_masked)
                else:
                    if len(_target_sent_embed) == 0 or _target_sent_embed[-1] != mask_tokens[masked_spans_counter]:
                        _target_sent_embed.append(mask_tokens[masked_spans_counter])
                i += 1
            masked_input_ids.append(_input_sent_embed)
            target_ids.append(_target_sent_embed)
        return masked_input_ids, target_ids

@dataclass
class MyDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel]
    max_source_length: int = 1024
    max_target_length: int = 128
    label_pad_token_id: int = -100
    pp: bool = True
    train_type_list: list = None
    input_aug: bool = False
    disturb: bool = False
    outdir: str = None

    def __call__(self, batch):
        # exit()
        # pdb.set_trace()
        # 'decoder_input_ids': 0,
        if self.disturb:
            # print(f"ori batch 11 {batch}", flush=True)
            pad_dict = {'input_ids': 0,
                        'attention_mask': 0,
                        'labels': self.label_pad_token_id,
                        'disturb_input_ids':0,
                        'disturb_attention_mask': 0}
            disturbed = disturb_batch(deepcopy(batch))
            # print(f"ori batch 22 {batch}", flush=True)
            # print(f"disturb batch {disturbed}", flush=True)
        else:
            pad_dict = {'input_ids': 0,
                        'attention_mask': 0,
                        'labels': self.label_pad_token_id,}
        features = {}
        lm_batch, mlm_batch, ground_batch, uttr_batch = None, None, None, None
        ground_disturbed_batch = None
        mixtwo_batch = None

        for pretrain_type in self.train_type_list:
            # * Process document-grounded dialogue generation.
            # logger.info(pretrain_type)

            if pretrain_type == 'add_responding':
                lm_batch = get_tokenized_batch(batch, pretraintype='add_responding', tokenizer=self.tokenizer, pp=self.pp, task_type='responding',max_source_length=self.max_source_length, 
                                                max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                if lm_batch is None: continue
                if self.disturb:
                    dist_lm_batch = get_tokenized_batch(disturbed, pretraintype=pretrain_type, tokenizer=self.tokenizer, pp=self.pp, task_type='disturbed_responding',
                                                        max_source_length=self.max_source_length, max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                    dist_lm_batch = {'disturb_'+k:v for k, v in dist_lm_batch.items()}
                    lm_batch.update(dist_lm_batch)

            # * Process lm response input and labels
            elif pretrain_type == 'add_understanding':
                T5_datacollator = DataCollatorForT5MLM(tokenizer=self.tokenizer, pad_token_id=self.label_pad_token_id,
                                                        model=self.model, decoder_start_token_id=self.tokenizer.pad_token_id,)
                mlm_features = T5_datacollator(batch)
                mlm_input_ids = mlm_features['input_ids']
                mlm_attn_mask = mlm_features['attention_mask']
                mlm_target = mlm_features['labels']
                mlm_nli_weights = mlm_features['nli_weights']
                mlm_target[mlm_target==self.tokenizer.pad_token_id] = self.label_pad_token_id
                mlm_batch = {'input_ids':mlm_input_ids,'attention_mask':mlm_attn_mask,'labels':mlm_target, 'nli_weights':mlm_nli_weights, 'batch':mlm_features['batch']}

                if self.disturb:
                    dist_mlm_features = T5_datacollator(disturbed)
                    dist_mlm_batch = {'disturb_input_ids':dist_mlm_features['input_ids'], 'disturb_attention_mask': dist_mlm_features['attention_mask'], }
                    mlm_batch.update(dist_mlm_batch)

            # * Process mlming generation.
            elif pretrain_type == 'add_grounding':
                ground_batch = get_tokenized_batch(batch, pretraintype=pretrain_type, tokenizer=self.tokenizer, pp=self.pp, task_type='grounding',max_source_length=self.max_source_length, 
                                                max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                if ground_batch is None: continue
                if self.disturb:
                    dist_ground_batch = get_tokenized_batch(disturbed, pretraintype=pretrain_type, tokenizer=self.tokenizer, pp=self.pp, task_type='disturbed_grounding',max_source_length=self.max_source_length, 
                                                            max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                    dist_ground_batch = {'disturb_'+k:v for k, v in dist_ground_batch.items()}
                    ground_batch.update(dist_ground_batch)
            
            elif pretrain_type == 'mix_twotasks':
                mixtwo_batch = get_tokenized_batch(batch, pretraintype=pretrain_type, tokenizer=self.tokenizer, pp=self.pp, task_type='mix_tasks',max_source_length=self.max_source_length, 
                                                 max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                if mixtwo_batch is None: continue
                if self.disturb:
                    dist_mix_batch = get_tokenized_batch(disturbed, pretraintype=pretrain_type, tokenizer=self.tokenizer, pp=self.pp, task_type='disturb_mix_tasks',max_source_length=self.max_source_length, 
                                                        max_target_length=self.max_target_length, label_pad_token_id=self.label_pad_token_id, outdir=self.outdir)
                    dist_mix_batch = {'disturb_'+k:v for k, v in dist_mix_batch.items()}
                    mixtwo_batch.update(dist_mix_batch)

        # pdb.set_trace()
        # * Pad data after different processes.
        # sen_list = ['input_sen','output_sen']
        batch_list = [lm_batch, mlm_batch, ground_batch, mixtwo_batch]
        # print(lm_batch.keys(), mlm_batch.keys(), ground_batch.keys())
        batch_list = [i for i in batch_list if i is not None]
        features = {}
        for feature_name, pad_value in pad_dict.items():
            max_length = max([i[feature_name].size()[1] for i in batch_list])
            newbatch_list = []
            for batch in batch_list:
                btz, maxl = batch[feature_name].size()
                # print(f'feature_name {feature_name}, btz {btz}', flush=True)
                newbatch = torch.cat((batch[feature_name], torch.full((btz, max_length-maxl), pad_value)), dim=1)
                newbatch_list.append(newbatch) 
            # logger.info(f'new batch size {[i.size() for i in newbatch_list]}')
            features[feature_name] = torch.cat(newbatch_list, dim=0)
        # print(f"FEATURE SIZES: {features['input_ids'].size(), features['labels'].size(), features['attention_mask'].size()}", flush=True)
        # logger.info(f"FEATURE per sample: {features['input_ids'][0,:], features['labels'][0,:], features['attention_mask'][0,:]}")

        # other_features = ['nli_weights', 'batch']  # add 'batch' for generation
        other_features = ['nli_weights', 'batch']  # add 'batch' for generation
        # other_features = ['nli_weights'] 
        for aa in other_features:
            # logger.info(f'aa {aa}')
            newbatch_list = []
            for batch in batch_list:
                if aa in batch:
                    newbatch_list += batch[aa]
                    # logger.info(f'newbatch {type(newbatch_list), newbatch_list}')
                    # features[aa] = torch.tensor(newbatch_list)
                features[aa] = newbatch_list
            # logger.info(f'pass')
        # features['typelist'] = typelist
        self.pp = False
        return features

   
def disturb_batch(batch):
    insert = random.choice([0,1])  #* random insert or delete
    # insert = 0 # only delete
    only_disturb_wiki = True #* not disturb reddit paraphrased data.
    # only_disturb_wiki = False

    btz = len(batch)
    newbatch = []

    insert_num = 1
    # insert_num = 3
    #* Insert some unrelated sentences into the original document
    for i in range(btz):
        sample = batch[i]        
        if sample['datatype'] == 'reddit':
            if only_disturb_wiki:
                newbatch.append(sample)
                continue
        elif sample['datatype'] == 'disturbwiki':
            insert = 0

        if insert:
            doc = sample['document']
            grd = sample['grounding']
            ori_sens = sent_tokenize(doc)

            other_doc = ' '.join([batch[pp]['document'] for pp in range(btz) if pp!= i])
            # print(f'other doc {other_doc}', flush=True)

            other_sens = sent_tokenize(other_doc)
            insert_num = min(insert_num, len(ori_sens)-1, len(other_sens))
            ss = random.sample(other_sens, insert_num)

            pos_list = random.sample(range(1,len(ori_sens)), insert_num)
            # print('insert position:',pos_list, flush=True)
            for k in range(len(ss)):
                ori_sens.insert(pos_list[k]+k, ss[k])

            sample['document'] = ' '.join(ori_sens)
            # print(f'ori doc {doc}', flush=True)
            # print(f"inserted doc {sample['document']}", flush=True)
            newbatch.append(sample)

        #* Delete some sentences from original document
        else:
            ori_doc = sample['document']
            doc = ori_doc[24:]
            grd = sample['grounding'][12:]

            start = doc.index(grd)
            end = start + len(grd)
            match = doc[start:end]
            if match != grd:
                num_wrong += 1
                print('False')
            assert match == grd        
            before = doc[:start]
            after = doc[end:]

            if sample['datatype'] == 'disturbwiki':    
                sample['document'] = '<title>Document</title> ' + before + after
            else:
                before_list = sent_tokenize(before)
                after_list = sent_tokenize(after)
                assert len(before_list)>=1 or len(after_list)>=1

                if len(before_list)>=1 and len(after_list)>=1:
                    del_be = random.choice([0,1]) 
                    if del_be:
                        del_idx = random.sample([k for k in range(0,len(before_list))], 1)[0]
                        before_list = before_list[:del_idx] + before_list[del_idx+1:]
                    else:
                        del_idx = random.sample([k for k in range(0,len(after_list))], 1)[0]
                        after_list = after_list[:del_idx] + after_list[del_idx+1:]
                elif len(before_list)<1:
                    del_idx = random.sample([k for k in range(0,len(after_list))], 1)[0]
                    after_list = after_list[:del_idx] + after_list[del_idx+1:]
                elif len(after_list)<1:
                    del_idx = random.sample([k for k in range(0,len(before_list))], 1)[0]
                    before_list = before_list[:del_idx] + before_list[del_idx+1:]

                sample['document'] = '<title>Document</title> '+ ' '.join(
                    before_list) + grd + ' '.join(after_list)
            newbatch.append(sample)

    if only_disturb_wiki and len(newbatch) == 0:
        return batch

    return newbatch


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]
