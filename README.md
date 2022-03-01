# SCD: Self-Contrastive Decorrelation of Sentence Embeddings
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!--- Register repository https://api.reuse.software/register, then add REUSE badge:
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/REPO-NAME)](https://api.reuse.software/info/github.com/SAP-samples/REPO-NAME)
-->

## Description
This repository will contain the source code for our paper *SCD: Self-Contrastive Decorrelation of Sentence Embeddings* to be presented at  [ACL2022](https://www.2022.aclweb.org/). The code is in parts based on the code from [Huggingface Tranformers](https://github.com/huggingface/transformers) and the paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE).

### Abstract
In this paper, we propose Self-Contrastive Decorrelation, a self-supervised approach, which takes an input sentence and optimizes a joint self-contrastive and decorrelation objective, with only standard dropout. This simple method works surprisingly well, achieves comparable results with state-of-the-art methods on multiple benchmarks without using contrastive pairs. This study opens up avenues for efficient self-supervised learning methods that are more resilient to train on a small batch regime than current contrastive methods.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/) (version 3.6 or later)
- [PyTorch](https://pytorch.org/)


## Download and Installation

## Known Issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2022 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
