python /home/jagan/Brucewan/MemoryBert/MemRoberta/mem_roberta/run_pretrained.py \
--output_dir=/home/jagan/Brucewan/MemoryBert/MemRoberta/models/chemprot_TAPT --model_type=roberta  --overwrite_output_dir \
--model_name_or_path=roberta-base --train_data_file=/home/jagan/Brucewan/MemoryBert/MemRoberta/data/chemprot/train.tsv \
--eval_data_file=/home/jagan/Brucewan/MemoryBert/MemRoberta/data/chemprot/dev.tsv --mlm --line_by_line \
--Ngram_path /home/jagan/Brucewan/MemoryBert/MemRoberta/ngram/chemprot/pmi_chemprot_ngram.txt --num_train_epochs 10.0 \
--learning_rate 4e-5


python /home/jagan/Brucewan/MemoryBert/MemRoberta/mem_roberta/run_finetune.py \
--model_name_or_path ./models/chemprot_TAPT \
--task_name ag --max_seq_length 256 --per_device_train_batch_size 16 \
--learning_rate 2e-5 --num_train_epochs 5.0 --output_dir ./results/ag_TAPT_FT/ \
--data_dir ./data/AG/ --Ngram_path /home/jagan/Brucewan/new_T-DNA/T-DNA/ngram/chemprot/pmi_chemprot_ngram.txt --overwrite_output_dir --save_steps 5000