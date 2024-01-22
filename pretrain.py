from dataset import Doc2DialDataset, MixedDataset, PinnedBatch, MyDataCollator
from pseudo_dataset import PseudoDataset
from logger import Logger
from myutils import *
from settings import *
from arguments import *
from mytrainer import MyTrainer
from mymodel import MyModel
import json
import os
import sys
import time, traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import datasets, transformers
from datasets import load_dataset, load_metric
from tqdm import tqdm
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed)
from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import torch, logging, math
import pickle
print(f'TORCH CUDA:', torch.cuda.is_available(), flush=True)

#* Setup logging
logger = Logger()

log_level=logging.INFO

def main():
    send_example_telemetry("run_t5_mlm", model_args, data_args)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data Arguments: {data_args}")
    logger.info(f"Model Arguments: {model_args}")
    logger.info(f'My Arguments: {myargs}')

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror='tuna',
    )


    #* Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_max_length=data_args.max_source_length
    )

    # model = AutoModelForSeq2SeqLM.from_pretrained(
    model = MyModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,)

    special_tokens = ["<last_turn>", "<user>", "<agent>", "<grounding>","<title>","</title>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    logger.info("Begin Data Preprocessing ...")
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    reddit_dataset, wiki_dataset, reddit_wo_evidence, wiki_wo_evidence = None, None, None, None
    wiki_disturb = None
    wikitrain_dataset = None

    if training_args.do_train:

        datapicklefile = os.path.join("./DocDATA", myargs.dataid+'.pt')
        if os.path.isfile(datapicklefile):
            logger.info(f'Load from {datapicklefile}')
            with open(datapicklefile, "rb") as fr:
                alltrain_dataset = pickle.load(fr)

        else:
            if myargs.add_redditdata:
                if myargs.use_pseudo_data:
                    reddit_dataset = PseudoDataset(myargs.reddit_pseudo_train_file, tokz=tokenizer, datatype='reddit', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)
                else:
                    reddit_dataset = Doc2DialDataset(data_files['train'], tokz=tokenizer, datatype='reddit', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)
                if myargs.add_reddit_wo_evidence:
                    reddit_wo_evidence = PseudoDataset(myargs.reddit_wo_evidence_train_file, tokz=tokenizer, datatype='reddit', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)

            if myargs.add_wikidata:
                wikipath = myargs.wikidialog_file
                if myargs.use_pseudo_data:
                    wikitrain_dataset = PseudoDataset(myargs.wiki_pseudo_train_file, tokz=tokenizer, datatype='wikidialog', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)
                else:
                    wikitrain_dataset = Doc2DialDataset(wikipath, tokz=tokenizer, datatype='wikidialog', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)
                if myargs.add_wiki_wo_evidence:
                    wiki_wo_evidence = PseudoDataset(myargs.wiki_wo_evidence_train_file, tokz=tokenizer, datatype='wikidialog', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)
                if myargs.unlikelihood:
                    wiki_disturb = PseudoDataset(myargs.wiki_disturb_train_file, tokz=tokenizer, datatype='disturbwiki', outdir=training_args.output_dir, num_data=myargs.num_data, myargs=myargs)

            assert reddit_dataset is not None or wikitrain_dataset is not None
            alltrain_dataset = MixedDataset([reddit_dataset, wikitrain_dataset, reddit_wo_evidence, wiki_wo_evidence, wiki_disturb], tokz=tokenizer)

            try:
                datapicklefile = os.path.join("./DocDATA", myargs.dataid+'.pt')
                if not os.path.isfile(datapicklefile):
                    logger.info(f'Process data to {datapicklefile}')
                    with open(datapicklefile, "wb") as f:
                        pickle.dump(alltrain_dataset, f)
            except:
                pass

    if training_args.do_eval:
        eval_dataset = Doc2DialDataset(data_files['validation'], tokz=tokenizer, outdir=training_args.output_dir, num_data=myargs.num_data)
        if myargs.add_wikidata:
            wiki_validpath = myargs.wikidialog_valid_file
            wikivalid_dataset = Doc2DialDataset(wiki_validpath, tokz=tokenizer, datatype='wikidialog', outdir=training_args.output_dir, num_data=myargs.num_data)
            alleval_dataset = MixedDataset([eval_dataset, wikivalid_dataset], tokz=tokenizer)
        else: alleval_dataset = eval_dataset

    training_args.remove_unused_columns = False
    logger.info("End Data Preprocessing ...")

    #* Initialize our Trainer
    logger.info(f'Initialize the trainer.')

    train_type_list = []
    if myargs.add_understanding: train_type_list.append('add_understanding')
    if myargs.add_grounding: train_type_list.append('add_grounding')
    if myargs.add_responding: train_type_list.append('add_responding')
    if myargs.add_uttr_generating: train_type_list.append('add_uttr_generating')
    if myargs.mix_twotasks: train_type_list.append('mix_twotasks')

    data_collator = MyDataCollator(model=model, tokenizer=tokenizer, max_source_length=data_args.max_source_length, 
                                   max_target_length=data_args.max_target_length, train_type_list=train_type_list,
                                   disturb=myargs.disturb, outdir=training_args.output_dir)
    # train_loader = DataLoader(alltrain_dataset, batch_size=4, num_workers=0, pin_memory=True, collate_fn=data_collator)
    # print(next(iter(train_loader)),flush=True)
    # a= next(iter(train_loader))

    if False:
        metric = load_metric("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=alltrain_dataset if training_args.do_train else None,
        eval_dataset=alleval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        myargs=myargs,
    )

    # TODO: Training ------------------------------------
    logger.info(f'Begin Training......')
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f'Run trainer...')
        # try:
            # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        # except Exception:
        #     traceback.print_exc()

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                alltrain_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(alltrain_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

if __name__ == "__main__":
    main()
