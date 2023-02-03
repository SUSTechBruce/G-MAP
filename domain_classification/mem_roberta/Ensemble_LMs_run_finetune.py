
""" Finetuning the library models for sequence classification on GLUE."""
# CUDA_VISIBLE_DEVICES=1,2,3  bash auto_TAPT.sh


import dataclasses
import logging
import os
from typing import Callable, Dict, List

import numpy as np


from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))

from configuration import AutoConfig
from Ensemble_LMs_trainer import EvalPrediction, Trainer, set_seed
from util import GlueDataset
from config import glue_compute_metrics, glue_tasks_num_labels
from tokenization import RobertaTokenizer
from Emsemble_LMs_modeling import RobertaForSequenceClassification


import torch
import argparse
from enum import Enum
import socket
from datetime import datetime
import csv

class EvaluationStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

logger = logging.getLogger(__name__)                    

class VocabDict(object):
    """
    Dict class to store the ngram
    """
    def __init__(self, ngram_freq_path, max_ngram_in_seq=20):
        """Constructs TDNANgramDict
        :param ngram_freq_path: ngrams with frequency
        """
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = []
        self.ngram_to_id_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram = line.strip()
                self.id_to_ngram_list.append(ngram)
                self.ngram_to_id_dict[ngram] = i

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram,freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Path to pretrained model or model identifier from huggingface.co/models")

    parser.add_argument("--Ngram_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to Ngram path")
    parser.add_argument("--config_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="Where do you want to store the pretrained models downloaded from s3")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        required=False,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir",
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--do_train",
                        action='store_false',
                        help="Whether to run training")
    parser.add_argument("--do_eval",
                        action='store_false',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_false',
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",
                        action='store_false',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--evaluation_strategy",
                        default="no",
                        type=EvaluationStrategy,
                        required=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--prediction_loss_only",
                        action='store_true',
                        help="When performing evaluation and predictions, only returns the loss.")
    parser.add_argument("--per_device_train_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_train_batch_size` is preferred. ")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_eval_batch_size` is preferred.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        required=False,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        required=False,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_beta1",
                        default=0.9,
                        type=float,
                        required=False,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2",
                        default=0.999,
                        type=float,
                        required=False,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        required=False,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        required=False,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        required=False,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        required=False,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        required=False,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_dir",
                        default="runs/" + datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname(),
                        type=str,
                        required=False,
                        help="Tensorboard log dir.")
    parser.add_argument("--logging_first_step",
                        action='store_true',
                        help="Log and eval the first global_step")
    parser.add_argument("--logging_steps",
                        default=500,
                        type=int,
                        required=False,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        default=500000,
                        type=int,
                        required=False,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit",
                        default=None,
                        type=int,
                        required=False,
                        help="Limit the total amount of checkpoints.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Do not use CUDA even when it is available")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        required=False,
                        help="random seed for initialization")
    parser.add_argument("--fp16",
                        action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        default="O1",
                        type=str,
                        required=False,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        required=False,
                        help="For distributed training: local_rank")
    parser.add_argument("--tpu_num_cores",
                        default=None,
                        type=int,
                        required=False,
                        help="TPU: Number of TPU cores (automatically passed by launcher script)")
    parser.add_argument("--tpu_metrics_debug",
                        action='store_true',
                        help="Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Whether to print debug metrics on TPU")
    parser.add_argument("--dataloader_drop_last",
                        action='store_true',
                        help="Drop the last incomplete batch if it is not divisible by the batch size.")
    parser.add_argument("--eval_steps",
                        default=None,
                        type=int,
                        required=False,
                        help="Run an evaluation every X steps.")
    parser.add_argument("--dataloader_num_workers",
                        default=0,
                        type=int,
                        required=False,
                        help="Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.")
    parser.add_argument("--past_index",
                        default=-1,
                        type=int,
                        required=False,
                        help="If >=0, uses the corresponding part of the output as the past state for next step.")
    parser.add_argument("--run_name",
                        default=None,
                        type=str,
                        required=False,
                        help="An optional descriptor for the run. Notably used for wandb logging.")
    parser.add_argument("--disable_tqdm",
                        action='store_true',
                        help="Whether or not to disable the tqdm progress bars.")
    parser.add_argument("--remove_unused_columns",
                        action='store_false',
                        help="Remove columns not required by the model when using an nlp.Dataset.")
    parser.add_argument("--label_names",
                        default=None,
                        type=List[str],
                        required=False,
                        help="The list of keys in your dictionary of inputs that correspond to the label")
    parser.add_argument("--general_model", default=None, type=str, required=True, help="general roberta model")
    parser.add_argument("--domain_model", default=None, type=str, required=True, help="domain roberta model")

    args = parser.parse_args()


    if args.no_cuda:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    elif args.local_rank == -1:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    try:
        num_labels = glue_tasks_num_labels[args.task_name]
        output_mode = "classification"
    except KeyError:
        raise ValueError("Task not found: %s" % (args.task_name))

    domain_vocab_dict = VocabDict(args.Ngram_path)
    # args.general_model + '/config.json'
    config = AutoConfig.from_pretrained(
        args.general_model + '/config.json' ,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
        Ngram_vocab_size=len(domain_vocab_dict.id_to_ngram_list),
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        args.general_model,
        cache_dir=args.cache_dir,
        Ngram_vocab_size=len(domain_vocab_dict.id_to_ngram_list),
    )


    domain_model_path = args.domain_model + '/pytorch_model.bin'

    domain_model = RobertaForSequenceClassification.from_pretrained(
        args.domain_model + '/pytorch_model.bin',
        from_tf=bool(".ckpt" in domain_model_path),
        config=config,
        cache_dir=args.cache_dir,
    )
    print('******************* Loading domain model success ! ***************************')


    # Loading model for general memory model
    general_model_path = args.general_model + '/pytorch_model.bin'
    general_model = RobertaForSequenceClassification.from_pretrained(
        args.general_model + '/pytorch_model.bin',
        from_tf=bool(".ckpt" in general_model_path),
        config=config,
        cache_dir=args.cache_dir
    )
    
    # mem_model.eval()

    # Get train dataset and test datasets

    train_dataset = (
        GlueDataset(args, domain_vocab_dict, tokenizer=tokenizer, cache_dir=args.cache_dir) if args.do_train else None
    )
    print(train_dataset)
    eval_dataset = (
        GlueDataset(args, domain_vocab_dict, tokenizer=tokenizer, mode="dev", cache_dir=args.cache_dir)
        if args.do_eval
        else None
    )
    print(eval_dataset)
    test_dataset = (
        GlueDataset(args, domain_vocab_dict, tokenizer=tokenizer, mode="test", cache_dir=args.cache_dir)
        if args.do_predict
        else None
    )
    
    print('execute here')
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer of domain model
    domain_trainer = Trainer(
        model=domain_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(args.task_name),

    )

    # Initialize our Trainer of mem model
    general_trainer = Trainer(
        model=general_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(args.task_name),

    )


    # Training
    if args.do_train:
        domain_trainer.train(
            model_path=domain_model_path
        )

        print('end of fine-tune domain model')
       
        if domain_trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output_dir)

        
        print('start to fine-tune general model')
        general_trainer.train(
            model_path='roberta-base'
        )
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)


    eval_results = {}
    
    if args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=args.cache_dir)
            )

        for test_dataset in test_datasets:
            domain_trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            general_model = general_trainer.output_general_model(eval_dataset=test_dataset) # 
            test_result = domain_trainer.evaluate(general_model, eval_dataset=test_dataset)
            metrics = test_result.metrics
            predictions = np.argmax(test_result.predictions, axis=1) + 1
            label_ids = test_result.label_ids + 1

            test_pred_file = os.path.join(
                args.output_dir, f"test_pred_{test_dataset.args.task_name}.txt"
            )
            f_test = open(test_pred_file, "w")
            f_test.write("input" + "\t" + "label" + "\t" + "pred" + "\n")

            with open(os.path.join(args.data_dir, "test.tsv"), "r", encoding="utf-8-sig") as f:
                input_labels = list(csv.reader(f, delimiter="\t"))
            for line, pred in zip(input_labels, predictions):
                text_a = line[0]
                label = line[1]
                f_test.write(text_a + '\t' + str(label) + '\t' + str(pred) + '\n')
            f_test.close()

            output_test_file = os.path.join(
                args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if domain_trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in metrics.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
