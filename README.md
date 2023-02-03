

# G-MAP

G-MAP: General Memory-Augmented Pre-trained Language Model for Domain Tasks



Source code for the EMNLP 2022 main conference paper entitled [G-MAP: General Memory-Augmented Pre-trained Language Model for Domain Tasks](https://arxiv.org/pdf/2212.03613.pdf)

Our experiment is build on the framework from [huggingface transformers](https://github.com/huggingface/transformers).4.4.1



## Files:
    .
    ├── domain_classification         # G-MAP Code for domain-specific classifcation tasks
    │   ├── data/...                  # domain-specific classification datasets
    │   ├── domain_models/...         # pretrained domain-specific models
    │   ├── mem_roberta/...           # main source code for classfication tasks
    │   ├── models/...                # pretrained general model
    │   └── output_model_dir/...      # saved output models 
    │          
    │
    ├── domain_qa_ner                 # G-MAP Code for domain-specific QA and NER tasks
    │   ├── data_ner/...              # domain-specific NER datasets
    │   ├── data_qa/...               # domain-specific QA datasets
    │   ├── domain_models/...         # pretrained domain-specific models    
    │   ├── mem_roberta/...           # main source code for QA and NER tasks
    |


To run the code for domain_document_classification tasks, the code is in ``domain_classification/``. This corresponds to the domain-specific classification experiments in the paper. 

To run the code for domain-specific Question Answering and NER tasks ,  the code is in ``domain_qa_ner``. This corresponds to the domain QA and NER experiments in the paper. 

The source codes are set to default of good hyperparameters, and can be used to train and inference for downstream-specific tasks :) 

-----------------------------------------------------
## Setup:

## Datasets

### Domain-specific Classification tasks

The experiments are conducted on eight classification tasks from four domains including biomedical sciences, computer science, news and reviews. They are: 
* **ChemProt**: a manually annotated chemical–protein interaction dataset extracted from 5,031 abstracts for relation classification;
* **RCT**: contains approximately 200,000 abstracts from public medicine with the role of each sentence clearly identified;
* **CitationIntent**: contains around 2,000 citations annotated for their function;
* **SciERC**: consists of 500 scientific abstracts annotated for relation classification; 
* **HyperPartisan**: which contains 645 articles from Hyperpartisan news with either extreme left-wing or right-wing stand-point used for partisanship classification;
* **AGNews**: consists of 127,600 categorized articles from more than 2000 news source for topic classification;
* **Amazon**:  consists of 145,251 reviews on Women’s and Men’s Clothing & Accessories, each representing users’ implicit feedback on items with a binary label signifying whether the majority of customers found the review helpful; 
* **IMDB**:  50,000 balanced positive and negative reviews from the Internet Movie Database for sentiment classification.

The domain-specific datasets can be downloaded and preprocessed from [the code associated with the Don't Stop Pretraining ACL 2020 paper](https://github.com/allenai/dont-stop-pretraining). Please cite this paper [**Gururangan et al. (2020)** ](https://arxiv.org/abs/2004.10964) if you use their datasets.

### Domain-specific QA tasks

In regard to the Medical QA task, please download the dataset from the below links https://github.com/panushri25/emrQA.

For the News QA task, please download the dataset from [NewsQA](https://drive.google.com/file/d/1TZCOm6lGKaz4fm_QaCrZladN-7YJkjt2/view?usp=sharing).

### Domain-specific NER tasks

Please check these datasets from https://huggingface.co/datasets

- conll2003

- ncbi_disease

- wnut_17

-----------------------------------------------------
## Training and inference example:

### Training of domain-specific document classification tasks

```python
cd domain_classification;
# Choose the tasks in run_main.py
python run_main.py --data_url /xxx --model_url --/xxx --save_dir --/xxx
```

### Training of domain-specific QA tasks

```python
cd domain_qa_ner;
# Choose the QA tasks in run_main.py
python run_main.py --data_url /xxx --model_url --/xxx --save_dir --/xxx

```

### Training of domain-specific NER tasks

```python
cd domain_qa_ner;
# Choose the NER tasks in run_main.py
python run_main.py --data_url /xxx --model_url --/xxx --save_dir --/xxx

```

## 

For details of the methods and results, please refer to our paper. 

```bibtex
@article{Wan2022GMAPGM,
  title={G-MAP: General Memory-Augmented Pre-trained Language Model for Domain Tasks},
  author={Zhongwei Wan and Yichun Yin and Wei Zhang and Jiaxin Shi and Lifeng Shang and Guangyong Chen and Xin Jiang and Qun Liu},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.03613}
}
```