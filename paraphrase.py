import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
import os
import logging
import time

datadir = './Reddit/data/final_data/'
outdir = './PRETRAIN/data/'

def get_response(input_list,num_return_sequences,num_beams):
    batch = tokenizer(input_list,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    # print(tgt_text)
    return tgt_text

if __name__=='__main__':
    logging.basicConfig(
      format="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] >> %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S")
    logger = logging.getLogger(__name__)
    logging.Formatter.converter = time.localtime
    logger.setLevel(logging.INFO)

    num_beams = 10
    num_return_sequences = 1
    # * No need to change-------------------
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'device {torch_device}')
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    # * ------------------------------------
    batch_size = 32

    # for ppp in range(0, 10):
    # for ppp in range(10, 20):
    # for ppp in range(20, 30):

    for ppp in range(1):
        # input_file = os.path.join(datadir,str(ppp)+'.json')
        input_file = os.path.join(datadir, '2021_selected.json')
        output_file = os.path.join(outdir, '2021_para.json')
        # output_file = os.path.join(outdir,str(ppp)+'_para.json')
        logger.info(f'Dealing with {input_file} NOW!')
        # get_response(ground_list, num_return_sequences,num_beams)    

        i = 0
        with open(output_file,'a') as fw:
            with open(input_file, 'r') as f:
                # data = [json.loads(i) for i in f.readlines()]
                k = 0
                sample_list = []
                ground_list = []
                for data in f.readlines():
                    aa = json.loads(data)

                    sample = json.loads(data)
                    ground = sample['response']
                    ground_list.append(ground)
                    sample_list.append(sample)
                    k += 1
                    if k == batch_size or (k<batch_size and i==1000000-1):
                        para_list = get_response(ground_list, num_return_sequences,num_beams)
                        for j in range(len(sample_list)):
                            # ss = {}
                            newsample = sample_list[j].copy()
                            # ss['original_text'] = sample_list[j]['response']
                            newsample['para_grounding'] = para_list[j] # btz>1
                            # ss['para_grounding'] = para_list # btz=1
                            # print(json.dumps(ss, ensure_ascii=False), file=fw)
                            print(json.dumps(newsample, ensure_ascii=False), file=fw)
                        k = 0
                        sample_list = []
                        ground_list = []
                        # break
                    i += 1
                    if i%100==0: 
                    #   break
                      logger.info(f'Have dealt with {i}-th sample.')
    logger.info('Done!')


# context = "The ultimate test of your knowledge is your capacity to convey it to another."
# get_response(context,num_return_sequences,num_beams)
# output:
# ['The test of your knowledge is your ability to convey it.',
#  'The ability to convey your knowledge is the ultimate test of your knowledge.',
#  'The ability to convey your knowledge is the most important test of your knowledge.',
#  'Your capacity to convey your knowledge is the ultimate test of it.',
#  'The test of your knowledge is your ability to communicate it.',
#  'Your capacity to convey your knowledge is the ultimate test of your knowledge.',
#  'Your capacity to convey your knowledge to another is the ultimate test of your knowledge.',
#  'Your capacity to convey your knowledge is the most important test of your knowledge.',
#  'The test of your knowledge is how well you can convey it.',
#  'Your capacity to convey your knowledge is the ultimate test.']

