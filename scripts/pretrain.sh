echo "$(date +'%r %a %d %h %y')"
now="$(date +'%m%d')"
port=5000
echo PORT $port
gpuid=0,1,2,3,4,5,6,7
ngpus=8 #!
echo GPUs $gpuid
num_data=2000000000

id=200w_wiki_reddit_NDE_TIE_run0
exp=${now}eflop_pretrain-$id 
dataid=reddit_wiki_pseudo
echo EXPERIMENT $exp
eflop=./PRETRAIN

# * Set python interpreter
py=python

# * Set hyperparameters
modelnamepath=./t5-large
modeltype=t5-large
# modeltype=t5-base

warmup_ratio=0.1

epochs=1

train_btz=1
eval_btz=2

grad_accum=8
eval=false
eval_steps=400
# modelname=t5-base
lr=1e-4
lr_scheduler=linear
mlm_pretrain=true
# mlm_pretrain=false
lm_generation=true
# lm_generation=false
# grounding=false
grounding=true
max_source_length=720

max_target_length=200
# data_workers=8
data_workers=0
# add_wiki=false
add_wiki=true
# add_reddit=false
add_reddit=true
# use_pseudo_data=true
use_pseudo_data=false
nli_selected=false
nli_weighted=false
# nli_selected=true
# nli_weighted=true
# nli_threshold=$3
# nli_threshold=0
nli_threshold=-1
# contrast=true
contrast=false
mix=true
local_rank=0
# fp16=true
fp16=false
#* whether disturb the input
disturb=true 
# disturb=false
alpha_kd=0.1
# KD_temperature=2.0
KD_temperature=1.0
# add_disturb_lm=true
add_disturb_lm=false
#* whether add kd_loss for NDE
add_disturb_kd=true
# add_disturb_kd=false
# whether add no evidence data
reddit_wo_evidence=false
# reddit_wo_evidence=true
wiki_wo_evidence=false
# wiki_wo_evidence=true
#* whether add unlikelihood loss for TIE
# unlikelihood=false 
unlikelihood=true

# ! Prepare the data files.--------------
# * Reddit Data
validdata=DATA/2122_reddit_para.json
reddit_pseudo_data=DATA/2122_reddit_para.json
# reddit_pseudo_data=DATA/10w_reddit_para.json
# data=DATA/10w_reddit_para.json
data=DATA/2122_reddit_para.json
# data=DATA/back500w_wikipara.json
reddit_wo_evidence_train_file=

# * Wikidialog Data
splitdir=./splitwiki
wiki_pseudo_data=DATA/100w_wikipara.json
wikidata=
wikivalid=
wiki_wo_evidence_train_file=
# wiki_disturb_train_file=DATA/100w_wikipara_disturb.json
wiki_disturb_train_file=DATA/100w_wikipara_disturb_new.json 
pretraindir=./DocOUT
mkdir -p $pretraindir ${pretraindir}/outputs ${pretraindir}/logs $pretraindir/logs/pretrain
output=${pretraindir}/outputs/pretrain/$exp
cache=${pretraindir}/cache
mkdir -p $output
log=${pretraindir}/logs/pretrain/$exp

echo "BEGIN PRETRAINING"
run=./torchrun

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=$gpuid \
$run --nproc_per_node=$ngpus --master_port=$port \
    pretrain.py \
    --local_rank=$local_rank \
    --output_dir=$output \
    --config_name=$modeltype \
    --tokenizer_name=$modeltype \
    --train_file=$data \
    --validation_file=$validdata \
    --do_train=true \
    --do_eval=$eval \
    --eval_steps=$eval_steps \
    --model_name_or_path=${modelnamepath} \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --overwrite_output_dir=true \
    --add_understanding=$mlm_pretrain \
    --add_responding=$lm_generation \
    --add_grounding=$grounding \
    --per_device_train_batch_size=$train_btz \
    --per_device_eval_batch_size=$eval_btz \
    --gradient_accumulation_steps=$grad_accum \
    --max_source_length=$max_source_length \
    --num_train_epochs=$epochs \
    --dataloader_num_workers=$data_workers \
    --report_to='tensorboard' \
    --cache_dir=$cache \
    --wikidialog_file=$wikidata \
    --wikidialog_valid_file=$wikivalid \
    --max_target_length=$max_target_length \
    --save_total_limit=50 \
    --save_steps=1000 \
    --warmup_ratio=$warmup_ratio \
    --add_wikidata=$add_wiki \
    --add_redditdata=$add_reddit \
    --num_data=$num_data \
    --use_pseudo_data=$use_pseudo_data \
    --wiki_pseudo_train_file=$wiki_pseudo_data \
    --reddit_pseudo_train_file=$reddit_pseudo_data \
    --nli_selected=$nli_selected \
    --nli_weighted=$nli_weighted \
    --nli_threshold=$nli_threshold \
    --contrast=$contrast \
    --mix_twotasks=$mix \
    --disturb=$disturb \
    --alpha_kd=$alpha_kd \
    --KD_temperature=$KD_temperature \
    --add_disturb_lm=$add_disturb_lm \
    --add_disturb_kd=$add_disturb_kd \
    --fp16=$fp16 \
    --add_reddit_wo_evidence=$reddit_wo_evidence \
    --add_wiki_wo_evidence=$wiki_wo_evidence \
    --reddit_wo_evidence_train_file=$reddit_wo_evidence_train_file \
    --wiki_wo_evidence_train_file=$wiki_wo_evidence_train_file \
    --unlikelihood=$unlikelihood \
    --wiki_disturb_train_file=$wiki_disturb_train_file \
    --dataid=$dataid \
    --deepspeed "configs/ds_new_config.json" \
    --bf16 True \
    >>$log.log 2>>${log}1.log
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap "T5Block"  \

echo "FINISH TRAINING"
