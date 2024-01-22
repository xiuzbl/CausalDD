import os,sys
import pdb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from eda import *
import torch


def apply_prompt(sample, type='add_responding',task=None, input_aug=False, reference=False):
    
    if type == 'add_understanding':
        if task == 'understand_dialog':
            all_sen = 'understand dialogue: ' + sample['utterance'] +' ' + sample['context']
        elif task == 'understand_document':
            all_sen = 'understand document: ' + sample['document']
        res = [{'input': all_sen}]

    elif type == 'add_responding':
        if not reference:
            input_sen = 'generate <agent>: ' + sample['utterance'] + ' ' + sample['context'] + ' ' + sample['document']
        else:
            input_sen = 'generate <agent>: ' + sample['utterance'] + ' ' + sample['context']

        output_sen = sample['response']
        if not input_aug:
            res = [{'input':input_sen, 'output': output_sen}]
        elif (not reference) and sample['datatype']=='reddit' and input_aug:
            # print(sample['utterance'],flush=True)
            # print(sample['utterance'][12:],flush=True)            
            uttr_aug_list = eda(sample['utterance'][12:], num_aug=3)
            doc_aug_list = eda(sample['document'][24:], num_aug=3)
            if not (uttr_aug_list and doc_aug_list): return None
            res = []
            for i in range(3):
                uttr = '<last_turn> ' + uttr_aug_list[i]
                doc = '<title>Document</title> ' + doc_aug_list[i]
                aug_input = 'generate <agent>: ' + uttr + ' ' + sample['context'] + ' ' + doc
                res.append({'input': aug_input, 'output':output_sen})
        else:
            res = [{'input':input_sen, 'output': output_sen}]

    elif type == 'add_grounding' and sample['datatype'] == 'wikidialog':
        input_sen = 'generate <grounding>: ' + sample['utterance'] + ' ' + sample['context'] + ' ' + sample['document']
        output_sen = sample['grounding']
        if not input_aug:
            res = [{'input':input_sen, 'output': output_sen}]
        elif (not reference) and sample['datatype']=='reddit' and input_aug:
            uttr_aug_list = eda(sample['utterance'][12:], num_aug=3)
            doc_aug_list = eda(sample['document'][24:], num_aug=3)
            if not (uttr_aug_list and doc_aug_list): return None
            res = []
            for i in range(3):
                uttr = '<last_turn> ' + uttr_aug_list[i]
                doc = '<title>Document</title> ' + doc_aug_list[i]
                aug_input = 'generate <grounding>: ' + uttr + ' ' + sample['context'] + ' ' + doc
                res.append({'input': aug_input, 'output':output_sen})
        else:
            res = [{'input':input_sen, 'output': output_sen}]
    
    elif type == 'add_uttr_generating' and (sample['datatype'] == 'downstream' or sample['datatype'] == 'reddit'):
        ori_context = sample['context']
        history = re.findall(r'<agent> (.*)', ori_context)
        if len(history):
            history = '<agent> ' + history[0]
        else: history = ""

        input_sen = 'generate <last_turn>: ' + history + ' ' + sample['document']
        output_sen = sample['utterance'] # <last_turn> + user_sentence

        res = [{'input':input_sen, 'output': output_sen}]
    
    elif type == 'mix_twotasks':
        input_sen = 'generate <grounding> then <agent>: ' + sample['utterance'] + ' ' + sample['context'] + ' ' + sample['document']
        output_sen = sample['grounding'] + ' ' + sample['response']
        res = [{'input':input_sen, 'output': output_sen}]

    else:
        return None

    return res

def printdata(batch, tokz, output_dir):
    btz = min(len(batch['input_ids']),len(batch['labels']))
    # outdir = training_args.output_dir
    outfile = open(os.path.join(output_dir, 'printed_data.log'),'a')

    for i in range(0, btz, 1):
        # sample = batch[i]
        print(f"{'*'*20} For sample {i} {'*'*20}",file=outfile)
        print(f"{'-'*10} Input ids {'-'*10}",file=outfile)
        print(batch['input_ids'][i],file=outfile)
        input_sen = tokz.decode(batch['input_ids'][i], file=outfile)
        print(input_sen, file=outfile)
        print(f"{'-'*10} Attention Mask {'-'*10}",file=outfile)
        print(batch['attention_mask'][i],file=outfile)
        print(f"Size of input ids is {batch['input_ids'][i].size()}",file=outfile)
        print(f"Size of input ids is {batch['attention_mask'][i].size()}",file=outfile)

        print(f"{'-'*10} Label ids {'-'*10}",file=outfile)
        print(batch['labels'][i],file=outfile)
        labels = batch['labels'][i]
        labels[labels==-100] = 0
        # pdb.set_trace()
        label_sen = tokz.decode(labels, file=outfile)
        print(label_sen, file=outfile)



def communicate_tensor(tensor_list, pad_token=0):
    '''
    collect tensors from all processes
    '''
    if len(tensor_list) == 0:
        return None
    device = tensor_list[0].device
    max_len = torch.tensor(max([i.shape[1] for i in tensor_list]), dtype=torch.int64, device=device)
    if dist.is_initialized():  # Obtain the max_len of the second dim of each tensor
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
    # Pad tensors to the max_len
    tensor = torch.cat([pad_tensor(i, max_len, pad_token) for i in tensor_list], dim=0)
    tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    max_tensor_bs = torch.tensor(tensor.shape[0], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(max_tensor_bs, op=dist.ReduceOp.MAX)  # Obtain the max_tensor_bs of each tensor
        if max_tensor_bs != tensor_bs:
            tensor = torch.cat([tensor, tensor.new(max_tensor_bs-tensor_bs, tensor.shape[1]).fill_(pad_token)], dim=0)

        # Gather padded tensors and the bs of each tensor
        tensor_list = [torch.ones_like(tensor).fill_(pad_token) for _ in range(dist.get_world_size())]
        tensor_bs_list = [torch.ones_like(tensor_bs).fill_(pad_token) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=tensor_list, tensor=tensor.contiguous())
        dist.all_gather(tensor_list=tensor_bs_list, tensor=tensor_bs)
        # Cut the padded batch
        for i in range(dist.get_world_size()):
            tensor_list[i] = tensor_list[i][:tensor_bs_list[i]]
        tensor = torch.cat(tensor_list, dim=0)
    return tensor

def cut_eos(seq, eos_id):
    if eos_id not in seq:
        return seq
    return seq[:seq.index(eos_id)]

def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class KDLoss(nn.Module):
    def __init__(self, alpha_kd=0.0, T=1.0):
        super(KDLoss, self).__init__()
        assert 0 <= alpha_kd <=5
        assert 0 < T
        self.alpha_kd = alpha_kd
        self.T = T
    
    def forward(self, output_logits, teacher_logits, label_mask=None):
        KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
            F.softmax(teacher_logits / self.T, dim=1), reduction='none')
        # print('label_mask shape:',label_mask.shape,'loss shape:',KD_loss.shape,flush=True)
        # KD_loss [batch_size*seq_len, 50528]

        KD_loss = torch.sum(KD_loss, dim=1)

        if label_mask is not None:
            label_mask = label_mask.view(-1)
            KD_loss = KD_loss.where(label_mask.cuda(), torch.tensor(0.0).cuda())
            kd_loss = KD_loss.sum() / label_mask.sum()

        else:
            kd_loss = KD_loss
        return kd_loss * self.alpha_kd * self.T * self.T

    # def forward(self, output_logits, teacher_logits=None):
    #     KD_loss = F.kl_div(F.log_softmax(output_logits / self.T, dim=1), 
    #         F.softmax(teacher_logits / self.T, dim=1), reduction='none')
    #     KD_loss = torch.sum(KD_loss, dim=1)
    #     return KD_loss * self.KD_term * self.T * self.T  # T^2 is a trick to make KL loss scale invariant to temperature

