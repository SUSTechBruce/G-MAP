# coding=utf-8

import os
import argparse
import fnmatch
import logging
import moxing as mox
import threading

DS_DIR_NAME = "src"
os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"

LOCAL_DIR = os.environ['DLS_LOCAL_CACHE_PATH']
assert mox.file.exists(LOCAL_DIR)
logging.info("local disk: " + LOCAL_DIR)
parser = argparse.ArgumentParser()

dir_name = os.path.dirname(os.path.abspath(__file__))

###### define args
parser.add_argument("--data_url", type=str, default="", required=True)
parser.add_argument("--model_url", type=str, default="", required=True)
parser.add_argument('--save_dir', type=str, default="/cache/src/output/", help="the path of saving traing model")

args, _ = parser.parse_known_args()


# copy data to local /cache/lcqmc
logging.info("copying data...")
local_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)

# copy data to local /cache/lcqmc
logging.info("copying data...")
local_data_dir = os.path.join(LOCAL_DIR, DS_DIR_NAME)
logging.info(mox.file.list_directory(args.data_url, recursive=True))
mox.file.copy_parallel(args.data_url, local_data_dir)

local_data_dir_output = os.path.join(local_dir, "output")
if not os.path.exists(local_data_dir_output):
    os.mkdir(local_data_dir_output)


print('Test the run main')
cmd1 ='''
   pip install tokenizers
   
   pip install regex
   
   python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 9588 mem_roberta/run_params_fuse_finetune_macro.py --task_name {task_name} --max_seq_length {max_seq_length} --per_device_train_batch_size {per_device_train_batch_size} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} --output_dir {output_dir} --data_dir {data_dir} --Ngram_path {Ngram_path} --general_model {general_model} --domain_model {domain_model} --seed {seed} --general_weight_alpha {general_weight_alpha} --domain_weight_alpha {domain_weight_alpha}
    '''.format(
        task_name='amazon',
        max_seq_length=256,
        per_device_train_batch_size=16,
        learning_rate=4e-5,
        num_train_epochs = 12,
        output_dir=local_data_dir_output,
        data_dir=os.path.join(local_data_dir, "amazon"), # chemprot, citation_intent, ag, amazon
        Ngram_path=local_data_dir+'/pmi_chemprot_ngram.txt',
        general_model=os.path.join(args.model_url, 'Roberta_base'),
        domain_model=os.path.join(args.model_url, 'Reviews'), # Chem_bio, CS, News, Reviews
        seed = 42,
        general_weight_alpha = 0.7,
        domain_weight_alpha = 0.3,
        )
# Domain: Chem_bio CS News Reviews 
# Task: chemprot rct-20k citation_intent sciie hyperpartisan_news ag amazon imdb
print(cmd1)
os.system(cmd1)


# copy output data
s3_output_dir = args.save_dir
logging.info("copy local data to s3")
logging.info(mox.file.list_directory(local_data_dir_output, recursive=True))
# s3_output_dir=os.path.join(os.path.join(args.data_url, task),args.output_dir)

print('output dir:' + s3_output_dir)
mox.file.copy_parallel(local_data_dir_output, s3_output_dir)