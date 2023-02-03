python /home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/mem_roberta/run_finetune.py \
--model_name_or_path roberta-base \
--task_name chemprot \
--max_seq_length 256 \
--per_device_train_batch_size 16 \
--learning_rate 4e-5 \
--num_train_epochs 12 \
--output_dir /home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/results/chemprot/ \
--data_dir /home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/data/chemprot \
--Ngram_path /home/wan/Desktop/Brucewan/video_scratch/Brucewan/MemoryBert/MemRoberta/ngram/chemprot/pmi_chemprot_ngram.txt \
--overwrite_output_dir

