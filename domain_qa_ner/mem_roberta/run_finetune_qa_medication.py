from ast import arg
import json
import logging
import os
import argparse
from collections import defaultdict, Counter
import random
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AutoTokenizer
logger = logging.getLogger(__name__)
# logging.disable(logging.WARNING)


import argparse
import os
import datetime
from trainer_qa import Trainer

from squad_metrics import (SquadResult, compute_predictions_logits,
                           squad_evaluate)

from  qa_medication_utils import (
    QAProcessor,
    convert_examples_to_features_mp,
)
from mem_atten_modeling import RobertaForQuestionAnswering
from roberta_modeling import RobertaModel

from configuration import AutoConfig


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def set_output_dir(args):
    basedir = "./qa_save"
    today = datetime.datetime.now().strftime("%Y%m%d")
    assert type(args.loc_layer) == str
    output_dir = f"{args.domain}"
    output_dir += f"_seed{args.seed}"
    return os.path.join(basedir, today, output_dir)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="/home/wan/Desktop/Brucewan/video_scratch/Brucewan/KALA/kala_datasets/medication_new")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=12)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_mention_length", type=int, default=30)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_model", type=str, default="roberta-based")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--version_2_with_negative", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/data_save/tmp")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--verbose_logging", action="store_true")
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--read_data", action="store_true",
                        help="read data from json file")

    parser.add_argument("--kala_learning_rate", type=float, default=3e-5)
    parser.add_argument("--loc_layer", type=str, default="11")
    parser.add_argument("--domain", type=str, default="NewsQA")
    parser.add_argument("--num_gnn_layers", type=int, default=2)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--general_model", default=None, type=str, required=True, help="general roberta model")
    parser.add_argument("--domain_model", default=None, type=str, required=True, help="domain roberta model")
    
    
    args = parser.parse_args()

    args.do_lower_case = True

    args.output_dir = set_output_dir(args)
    print(f"Output Directory: {args.output_dir}")

    args.loc_layer = [int(x) for x in args.loc_layer.split(',')]
    # args.data_dir = args.data_dir.replace("NewsQA", args.domain)
    print(f"Data Directory: {args.data_url}")

    args.pickle_folder = args.data_url
    print(f"KFM location: {args.loc_layer}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    return args



WEIGHTS_NAME = "pytorch_model.bin"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def initialize_roberta_model(args):
    print('############### Intializing the roberta model ################')
    config = AutoConfig.from_pretrained(
        args.general_model + '/config.json' , 
    )
    print("Show config: ", config)
    tokenizer = AutoTokenizer.from_pretrained( # using roberta-based
            args.general_model,
            use_fast=True,
            add_prefix_space=True
        )
    # Loading model for domain specific model
    domain_model_path = args.domain_model + '/pytorch_model.bin'
    
    
    model = RobertaForQuestionAnswering.from_pretrained(
        domain_model_path,
        from_tf=bool(".ckpt" in domain_model_path),
        config=config,
    )
    print('******************* Loading domain model success ! ***************************')

    # Loading model for general memory model

    mem_model = RobertaModel.from_pretrained(
        args.general_model + '/pytorch_model.bin',
        from_tf=bool(".ckpt" in 'roberta-base'),
        config=config,
    )
    args.tokenizer = tokenizer

    print('******************* Loading mem model success ! ***************************')

    return model, mem_model, tokenizer

    

def run(args):
    set_seed(args.seed)
    if args.local_rank == -1:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # entity_embeddings, wikidata_to_memory_map = load_entity_embeddings_memory(args)

    model, mem_model, tokenizer = initialize_roberta_model(args)

    model.to(args.device)
    mem_model.to(args.device)

    train_dataloader, _, _, _ = load_examples(args, "train")

    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

    best_dev_score = [-1]
    best_weights = [None]
    results = {}

    def step_callback(model, global_step):
        if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
            epoch = int(global_step / num_train_steps_per_epoch - 1)
            dev_results = evaluate(args, model, mem_model, fold="dev")   # modified here mem_model
            tqdm.write("dev: " + str(dev_results))
            results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
            if dev_results["f1"] > best_dev_score[0]:
                if hasattr(model, "module"):
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                else:
                    best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                best_dev_score[0] = dev_results["f1"]
                results["best_epoch"] = epoch
            model.train()

    if not args.do_eval: # True:
        trainer = Trainer(
            args,
            model=model,
            mem_model=mem_model,
            dataloader=train_dataloader,
            num_train_steps=num_train_steps,
            step_callback=step_callback
        )
        trainer.train()
        dev_results = evaluate(args, model, mem_model, fold="dev")
        print(dev_results)

        logger.info("Saving the model checkpoint to %s", args.output_dir)


    # Evaluate
    output_file = os.path.join(args.output_dir, "predictions.json")
    results = evaluate(args, model, mem_model, fold="test", output_file=output_file)
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    print(results)
    print(args.output_dir)
    return results

def evaluate(args, model, mem_model, fold="dev", output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    tokenizer = args.tokenizer

    all_results = []
    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        mem_model.eval()

        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():

            input_ids=inputs['input_ids']
            mem_attention_mask=inputs['attention_mask']
            token_type_ids=inputs['token_type_ids']
            output_attentions=True
            output_hidden_states=True

            mem_outputs = mem_model(input_ids=input_ids,
                                attention_mask=mem_attention_mask, 
                                token_type_ids=token_type_ids,
                                output_attentions=output_attentions, 
                                output_hidden_states=output_hidden_states)

            mem_states = [hidden_state for hidden_state in mem_outputs[2][:-1]]
            
     
            # setting the input of the domain_model
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            # start_positions = inputs['start_positions']
            # end_positions = inputs['end_positions']

            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    # start_positions=start_positions,
                    # end_positions=end_positions,
                    output_attentions=output_attentions, 
                    output_hidden_states=output_hidden_states,
                    mem_attention_mask=mem_attention_mask,
                    mem_states=mem_states)

            start_logits, end_logits = outputs[0], outputs[1]  #  modified here to fit the roberta model
            outputs = (start_logits, end_logits)


        feature_indices = batch["feature_indices"]
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output

            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )
    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results



def load_examples(args, fold):
    # wikidata_to_memory_map = args.wikidata_to_memory_map
    processor = QAProcessor(args)
    if fold == "train":
        examples = processor.get_train_examples(args.data_url)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_url)
    else:
        examples = processor.get_test_examples(args.data_url)


    features = convert_examples_to_features_mp(
            examples,
            args.tokenizer,
            args.max_seq_length,
            args.doc_stride,
            args.max_query_length,
            is_training=fold=="train")

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    

        ret = dict(
            input_ids=create_padded_sequence("input_ids", args.tokenizer.pad_token_id), # input of roberta
            attention_mask=create_padded_sequence("input_mask", 0), # input of roberta
            token_type_ids=create_padded_sequence("segment_ids", 0), # input of roberta
        
        )
        if fold == "train":
            ret["start_positions"] = torch.stack([torch.tensor(getattr(o[1], "start_position"), dtype=torch.long) for o in batch])
            ret["end_positions"] = torch.stack([torch.tensor(getattr(o[1], "end_position"), dtype=torch.long) for o in batch])
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)
        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:       
            sampler = DistributedSampler(features)
            
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn, num_workers=0
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=0)

    return dataloader, examples, features, processor

if __name__ == "__main__":
    args = setup_args()
    run(args)
