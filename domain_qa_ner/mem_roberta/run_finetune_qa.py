from ast import arg
import imp
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

from qa_utils_ import (
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
    parser.add_argument("--data_dir", type=str, default="/home/wan/Desktop/Brucewan/video_scratch/Brucewan/KALA/KGC/TASK/NewsQA")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--max_mention_length", type=int, default=30)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
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
    args = parser.parse_args()

    args.do_lower_case = True

    args.output_dir = set_output_dir(args)
    print(f"Output Directory: {args.output_dir}")

    args.loc_layer = [int(x) for x in args.loc_layer.split(',')]
    args.data_dir = args.data_dir.replace("NewsQA", args.domain)
    print(f"Data Directory: {args.data_dir}")

    args.pickle_folder = args.data_dir
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
    print('Intializing the roberta model ################')
    config = AutoConfig.from_pretrained(
        '/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/domain_models/Chem_bio/config.json' ,
    )
    print("Show config: ", config)
    tokenizer = AutoTokenizer.from_pretrained( # using roberta-based
            "roberta-base",
            use_fast=True,
            add_prefix_space=True
        )
    # Loading model for domain specific model
    chembio = '/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/domain_models/Chem_bio/pytorch_model.bin'
    cs = '/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/domain_models/CS/pytorch_model.bin'
    new = '/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/domain_models/News/pytorch_model.bin'
    review = '/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/domain_models/Reviews/pytorch_model.bin'
    domain_model_path = new
    model = RobertaForQuestionAnswering.from_pretrained(
        domain_model_path,
        from_tf=bool(".ckpt" in domain_model_path),
        config=config,
    )
    print('******************* Loading domain model success ! ***************************')

    # Loading model for general memory model

    mem_model = RobertaModel.from_pretrained(
        'roberta-base',
        from_tf=bool(".ckpt" in 'roberta-base'),
        config=config,
    )
    args.tokenizer = tokenizer

    print('******************* Loading mem model success ! ***************************')

    return model, mem_model, tokenizer

    

def run(args):
    set_seed(args.seed)
    args.device = 'cuda'

    entity_embeddings, wikidata_to_memory_map = load_entity_embeddings_memory(args)

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
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

        # Load the best model on validation set for evaluation
        model, _, tokenizer = initialize_roberta_model(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)

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

            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
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

def load_entity_embeddings_memory(args):
    memory_path = os.path.join(args.data_dir, "train_entity_embeddings.pkl")

    with open(memory_path, 'rb') as f:
        entity_embeddings_memory = pickle.load(f)
    wikidata_to_memory_map = dict()
    entity_embeddings = []

    for key, value in entity_embeddings_memory.items():
        wikidata_to_memory_map[key] = len(entity_embeddings) + 1
        entity_embeddings.append(value)

    entity_embeddings = torch.from_numpy(np.stack(entity_embeddings, axis=0))
    print(f"# Entity Embeddings: {len(entity_embeddings)}")
    args.entity_embed_size = entity_embeddings.shape[-1]
    # args.entity_embed_size = 768
    args.wikidata_to_memory_map = wikidata_to_memory_map

    entity_embeddings = torch.cat([torch.zeros(1, entity_embeddings.shape[-1]), entity_embeddings], dim=0)
    return entity_embeddings, wikidata_to_memory_map

def load_examples(args, fold):
    wikidata_to_memory_map = args.wikidata_to_memory_map
    processor = QAProcessor(args)
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    pickle_name = "train_features_bert.pkl"
    pickle_path = os.path.join('/home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/mem_roberta/qa_pkl', pickle_name)


    test = True
    if not os.path.exists(pickle_path) or (fold == "dev" or fold == "test") or args.read_data:
        print("Creating features from the dataset...")
        features = convert_examples_to_features_mp(
            examples,
            args.tokenizer,
            args.max_seq_length,
            args.doc_stride,
            args.max_query_length,
            is_training=fold=="train"
        )
        if fold == "train":
            with open(pickle_path, 'wb+') as f:
                pickle.dump(features, f)
    else:
        print("Loading cached features...")
        with open(pickle_path, 'rb') as f:
            features = pickle.load(f)

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        def retrieve(key):
            if key in wikidata_to_memory_map.keys():
                return wikidata_to_memory_map[key]
            else:
                return 0

        """ convert to torch_geometric batch type """
        batch_nodes = []
        batch_edge_index = []
        batch_edge_attr = []
        graph_batch = []
        batch_local_indicator = []
        for batch_idx, (_, item) in enumerate(batch):
            nodes = [retrieve(node) for node in item.wikidata_ids]
            edge_index = [[len(graph_batch) + edge[0], len(graph_batch) + edge[1]] for edge in item.edge_index]
            # Reverse (Bidirectional)
            rev_edge_index = []
            rev_edge_attr = []
            for edge, edge_attr in zip(item.edge_index, item.edge_attr):
                rev_edge = [len(graph_batch) + edge[1], len(graph_batch) + edge[0]]
                if rev_edge in edge_index:
                    continue
                rev_edge_index.append(rev_edge)
                rev_edge_attr.append(edge_attr)
            
            graph_batch += [batch_idx] * len(nodes)
            batch_nodes += nodes
            batch_edge_index += edge_index
            batch_edge_attr += item.edge_attr
            
            batch_edge_index += rev_edge_index
            batch_edge_attr += rev_edge_attr
            batch_local_indicator += item.local_indicator

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
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor

if __name__ == "__main__":
    args = setup_args()
    run(args)
