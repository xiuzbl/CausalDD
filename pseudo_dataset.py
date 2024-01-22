from myutils import *
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
from eda import get_only_chars
import nltk

nltk.data.path.append('./nltk')
from nltk.tokenize import sent_tokenize

np.set_printoptions(threshold=sys.maxsize)

def get_reddit_dict(sample, grounding=None):
    dialog_list = []
    response =  sample['response'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
    if 'turns' in sample: # for Reddit
        for user in sample['turns'].values():
            for sen in user.values():
                dialog_list.append(sen) 
        response = '<agent> ' + response

    context = get_context(dialog_list)    
    utterance = dialog_list[-1].replace('\n',' ').replace('\\',' ').replace('\t',' ')
    if not get_only_chars(utterance): return None
    utterance = '<last_turn> ' + utterance

    document = sample['related_passage'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
    if grounding is not None:
        doc_list = sent_tokenize(document)
        # doc_list = document.split('. ')
        # print(f'doc list: {doc_list}', flush=True)
        if len(doc_list) == 1:
            doc_list.insert(2, grounding)
        else:
            insert_idx = random.choice(range(1, len(doc_list)))
            doc_list.insert(insert_idx, grounding)
        document = ' '.join(doc_list)

    if len(document)<3 or not get_only_chars(document):
        document = None
    else:
        document = '<title>Document</title> ' + document
    return utterance, context, document, response


def print_sample(input_sen, output_sen=None, task_type=''):
    outfile=open(os.path.join('./','data_sample.log'),'a')
    print(f'-'*100, file=outfile)
    print(f'Task type: {task_type}', file=outfile)
    print(f'Input sentence:\n {input_sen}', file=outfile)
    if output_sen is not None: print(f'Output sentence:\n {output_sen}', file=outfile)
    print(f'*'*100, file=outfile)


def get_context(dialog_list):
    context_list = []
    odd = True if len(dialog_list) % 2 == 1 else False
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
    return context

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, max_source_length=1024, tokz=None, datatype='reddit', num_data=int(1e10), outdir=None, myargs=None):
        self.tokenizer = tokz
        self.data_path = data_path
        self.max_source_length = max_source_length
        self.type = datatype
        self.num_data = num_data
        self.outdir = outdir

        self.myargs = myargs

        if datatype == 'wikidialog':
            datadir = './data/splitwiki'
        elif datatype == 'disturbwiki':
            datadir = './data/splitdisturbwiki'
        self.data = []

        # if 'wiki' in datatype:
        #     num = 2
        #     for i in range(1,num):
        #         ddpath = os.path.join(datadir, str(i)+'.json')
        #         with open(ddpath, 'r') as f:
        #             for i in f:
        #                 sample = json.loads(i)
        #                 try:
        #                     ss = self.per_sample(sample, print_data=False)
        #                 except Exception:
        #                     ss = None
        #                 if ss is not None:
        #                     self.data.append(ss) 
        #             # dd = [json.loads(i) for i in f]
        #             # print(f'Finish loading the data locally from {wikipath}', flush=True)
        #         # self.data += self.data_process(dd)
        #         print(f'Finish processing data from {ddpath}', flush=True)
        # else:
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
        dialog_list = [] 

        if 'nli_score' in sample and self.myargs.nli_weighted:
            nli_weight = sample['nli_score']
        else:
            nli_weight = 1.0

        #todo For generated pseudo reddit------------------------------------------------
        if self.myargs.use_pseudo_data and self.type == 'reddit':
            if 'pseudo_label' in sample: 
                input_sen = sample['input_sen']
                response =  sample['pseudo_label'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
                utterance = re.findall(r'generate <agent>: (.*) <user>', input_sen)[0] #!error to fix
                document = re.findall(r'<title>Document</title> (.*)', input_sen)[0]
                document = '<title>Document</title> ' + document
                context = re.findall(r'<user> (.*) <title>Document', input_sen)[0]
                context = '<user> ' + context

            # * For generated pseudo utterance
            elif 'pseudo_utterance' in sample:
                response =  sample['response']
                utterance = sample['pseudo_utterance']
                document = sample['related_passage'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
                history = re.findall(r'<agent> (.*)', sample['context'])
                # print(f'history {history}',flush=True)
                if len(history):
                    history = '<agent> ' + history[0]
                else: history = ""
                uttr = re.findall(r'<last_turn>(.*)', utterance)[0] 
                context = '<user>' + uttr + history
            
            elif 'para_grounding' in sample:
                # print(f'Run this line.', flush=True)
                grounding = sample['para_grounding'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
                utterance, context, document, response = get_reddit_dict(sample, grounding=grounding)
                # utterance, context, document, response = get_reddit_dict(sample, grounding=None) #* for analysis test

                if document is None:
                    # print(f'document is None!!!', flush=True)
                    return None
                grounding = '<grounding> ' + grounding

        #todo For constructed wikidialog-------------------------------------------------------
        elif self.myargs.use_pseudo_data and (self.type == 'wikidialog' or self.type == 'disturbwiki'):

            #* Get paraphrased response
            if 'para_response' in sample:
                response =  sample['para_response'].replace('\n',' ').replace('\\',' ').replace('\t',' ')  # paraphrased response is the pseudo response
            else:
                response =  sample['response'].replace('\n',' ').replace('\\',' ').replace('\t',' ')  # paraphrased response is the pseudo response
            response = '<agent> ' + response

            #* Get grounding
            grounding = sample['response'].replace('\n',' ').replace('\\',' ').replace('\t',' ') # real response is the grounding text
            grounding = '<grounding> ' + grounding

            dialog_list = sample['utterances']
            context = get_context(dialog_list)    

            utterance = dialog_list[-1].replace('\n',' ').replace('\\',' ').replace('\t',' ')
            # if not get_only_chars(utterance): return None
            utterance = '<last_turn> ' + utterance

            document = sample['related_passage'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
            # if len(document)<3 or not get_only_chars(document):
            if len(document)<3:
                return None

            document = '<title>Document</title> ' + document

        #todo For real samples from Reddit and WikiDialog------------------------------------------------------------
        elif 'turns' in sample or 'utterances' in sample and self.type == 'reddit' and not self.myargs.use_pseudo_data:
            response =  sample['response'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
            grounding = ''
            if 'turns' in sample: # for Reddit
                for user in sample['turns'].values():
                    for sen in user.values():
                        dialog_list.append(sen) 
                response = '<agent> ' + response
            elif 'utterances' in sample and self.type == 'wikidialog': # for wikipedia
                dialog_list = sample['utterances']
                response = '<grounding> ' + response #? old implementation for wikidialog

            # if len(dialog_list) == 1:
            #     history = ""
            # else:
            #     history = ' '.join(dialog_list[:-1]).replace('\n',' ').replace('\\',' ').replace('\t',' ')

            context = get_context(dialog_list)    
            utterance = dialog_list[-1].replace('\n',' ').replace('\\',' ').replace('\t',' ')
            # if not get_only_chars(utterance): return None
            utterance = '<last_turn> ' + utterance

            # document = ' '.join(sample['related_passage']).replace('\n',' ').replace('\\',' ').replace('\t',' ')
            document = sample['related_passage'].replace('\n',' ').replace('\\',' ').replace('\t',' ')
            # if len(document)<3 or not get_only_chars(document):
            if len(document)<3:
                return None
            document = '<title>Document</title> ' + document

        res_dict = {
            'context': context,
            'utterance': utterance, 
            'response': response,
            'grounding': grounding, 
            'document': document,
            'datatype': self.type,
            'nli_weight': nli_weight,
        }

        # print(f'sample dict {res_dict}', flush=True)
        # print(f'Per sample processing done.', flush=True)
        if print_data and self.outdir:
            outfile=open(os.path.join(self.outdir,'printed_data.log'),'a')
            print(f'-'*100, file=outfile)
            print(f'Data Type {self.type}', file=outfile)
            print(f'Sample:\n {json.dumps(sample, ensure_ascii=False)}', file=outfile)
            print(f'Returned_dictionary:\n {json.dumps(res_dict,ensure_ascii=False)}', file=outfile)
            print(f'*'*100, file=outfile)

        return res_dict

    def data_process(self, data):
        printdata = True
        outdata = []
        # print(f'Len data: {len(data)}', flush=True)
        k=0
        for i in data:
            if i is not None:
                if 'nli_score' in i:
                    nli_score = i['nli_score']
                    if nli_score < self.myargs.nli_threshold:
                        # print('continue nli<0', flush=True)
                        continue
                k += 1
                # print(f'Begin per sample processing', flush=True)
                try:
                    sample = self.per_sample(i, print_data=printdata)
                except Exception:
                    sample = None
                if sample is not None:
                    outdata.append(sample)
                printdata = False
        # print(f'k: {k}', flush=True)
        return outdata    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # print(f'index {index}', flush=True)
        return self.data[index]

class MixedDataset(PseudoDataset):
    def __init__(self, datalist, max_source_length=1024, tokz=None):
        self.tokenizer = tokz
        self.max_source_length = max_source_length

        self.data = []
        for dataset in datalist:
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
        input_sen = [apply_prompt(i, type='add_understanding', task='understand_dialog')['input'] for i in batch]
        input_sen += [apply_prompt(i, type='add_understanding', task='understand_document')['input'] for i in batch]
        # input_encoding = self.tokenizer(input_sen, max_length=args.max_source_length, truncation=True, return_tensors='pt', padding='longest') 
        # if self.pp: print_sample(input_sen[0], output_sen[0], task_type='understanding')
        input_ids = [self.tokenizer.encode(i, max_length=self.max_length, truncation=True, return_tensors='pt') for i in input_sen]
        # logger.info(f'{input_sen[0]}')
        # logger.info(f'{input_ids[0]}')

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
            'labels': torch.tensor(labels).contiguous(),
            'batch': batch}
        features['attention_mask'] = features['input_ids'] > 0
        self.pp = False
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

