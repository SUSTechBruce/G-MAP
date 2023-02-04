print(1)
import jsonlines
import os
import csv

tasks = ['imdb']
PATH_TO_DONT_STOP_DATA = "/home/jagan/Brucewan/MemoryBert/MemRoberta/data"
PATH_TO_SAVE_DATA = "/home/jagan/Brucewan/MemoryBert/MemRoberta/data"
for task in tasks:
    target_path = f"{PATH_TO_SAVE_DATA}/{task}/"
    print('target_path', target_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    f_write = open(f"{target_path}/train.tsv", "w")
    tsv_w = csv.writer(f_write, delimiter='\t')

    labels = set()
    with open(f'{PATH_TO_DONT_STOP_DATA}/{task}/train.jsonl', 'rb') as f:
        try:
            for item in jsonlines.Reader(f):
                item['text'] = item['text'].replace('\n', ' ')  #only for amazon and imdb because there are \n in amazon and imdb
                tsv_w.writerow([item['text'], item['label']])
                labels.add(item['label'])
        except:
            print('Some data error')
    print('all labels: ', labels)
    print('number of labels: ', len(labels))
    f_write.close()