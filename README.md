# SCD: Self-Contrastive Decorrelation of Sentence Embeddings
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/acl2022-self-contrastive-decorrelation)](https://api.reuse.software/info/github.com/SAP-samples/acl2022-self-contrastive-decorrelation)


#### News
- **03/23/2022:** Provided source code
- 03/16/2022:  Added arXiv pre-print, provided models for download

## Description
This repository **will contain** the source code for our paper [**SCD: Self-Contrastive Decorrelation of Sentence Embeddings**](http://arxiv.org/abs/2203.07847) to be presented at [ACL2022](https://www.2022.aclweb.org/). The code is in parts based on the code from [Huggingface Tranformers](https://github.com/huggingface/transformers) and the paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE).

### Abstract
In this paper, we propose Self-Contrastive Decorrelation, a self-supervised approach, which takes an input sentence and optimizes a joint self-contrastive and decorrelation objective, with only standard dropout. This simple method works surprisingly well, achieves comparable results with state-of-the-art methods on multiple benchmarks without using contrastive pairs. This study opens up avenues for efficient self-supervised learning methods that are more resilient to train on a small batch regime than current contrastive methods.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/) (version 3.6 or later)
- [PyTorch](https://pytorch.org/)


## Download and Installation

1. Clone this repository
```
git clone https://github.com/SAP-samples/acl2022-self-contrastive-decorrelation scd
cd scd
```

2. Install the requirements

```
pip install -r requirements.txt
```

3. Download training data

```
cd data
sh download_wiki.sh
cd ..
```

4. Download evaluation dataset

```
cd SentEval/data/downstream/
sh download_dataset.sh
cd ../../../
```

## Language Models

Language models trained for which the performance is reported in the paper are available at the [Huggingface Model Repository](https://huggingface.co/models):
 - [BERT-base-uncased: sap-ai-research/BERT-base-uncased-SCD-ACL2022](https://huggingface.co/sap-ai-research/BERT-base-uncased-SCD-ACL2022)
 - [RoBERTa-base: sap-ai-research/RoBERTa-base-SCD-ACL2022](https://huggingface.co/sap-ai-research/RoBERTa-base-SCD-ACL2022)

Loading the model in Python. Just place in the model name as indicated above, e.g., sap-ai-research/BERT-base-uncased-SCD-ACL2022.

```shell
tokenizer = AutoTokenizer.from_pretrained("sap-ai-research/<----Enter Model Name---->")

model = AutoModelWithLMHead.from_pretrained("sap-ai-research/<----Enter Model Name---->")
```

With these models one should be able to reproduce the results on the benchmarks are reported in the paper:

For BERT-base-uncased:

```shell
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 66.94 | 78.03 | 69.89 | 78.73 | 76.23 |    76.30     |      73.18      | 74.19 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
+-------+-------+-------+-------+-------+-------+-------+-------+
|   MR  |   CR  |  SUBJ |  MPQA |  SST2 |  TREC |  MRPC |  Avg. |
+-------+-------+-------+-------+-------+-------+-------+-------+
| 79.26 | 84.85 | 93.57 | 89.13 | 85.23 | 83.60 | 74.84 | 84.35 |
+-------+-------+-------+-------+-------+-------+-------+-------+
```

For RoBERTA-base:

```shell
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 63.53 | 77.79 | 69.79 | 80.21 | 77.29 |    76.55     |      72.10      | 73.89 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
+-------+-------+-------+-------+-------+-------+-------+-------+
|   MR  |   CR  |  SUBJ |  MPQA |  SST2 |  TREC |  MRPC |  Avg. |
+-------+-------+-------+-------+-------+-------+-------+-------+
| 82.17 | 87.76 | 93.67 | 85.69 | 88.19 | 83.40 | 76.23 | 85.30 |
+-------+-------+-------+-------+-------+-------+-------+-------+
```

## Citations
If you use this code in your research or want to refer to our work, please cite:

```
@inproceedings{klein2022scd,
    title={SCD: Self-Contrastive Decorrelation for Sentence Embeddings},
     author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    month = may,
    year = "2022",
    publisher = "Association for Computational Linguistics (ACL)",
}
```


## Known Issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2022 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
