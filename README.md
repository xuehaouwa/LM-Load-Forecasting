# LM-Load-Forecasting


## Introduction
This folder includes the code and more details of our paper: Utilizing Language Models for Energy Load Forecasting, which will be presented at BuildSys 2023.
This work is developed from our previous work: [PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting](https://arxiv.org/abs/2210.08964)

Its code is also available [here.](https://github.com/HaoUNSW/PISA) 


## How to USE
1. prompt your data (describe your energy usage csv data to language sentences): use `data_prompting.py`
2. prepare train/val/test set data: use `prepare_hf.py`
3. fine-tune language models: use `run_hf_s2s.py`
4. test your fine-tuned language models with your test set: user `run_inference.py`

* For step 3 and 4, there are examples provided in `example.sh`
* Our PromptCast Repo also provides more details, you can check [here.](https://github.com/HaoUNSW/PISA) 



## Note

If you think our paper/code is useful, please cite our paper:
````
@inproceedings{10.1145/3600100.3623730,
author = {Xue, Hao and Salim, Flora D.},
title = {Utilizing Language Models for Energy Load Forecasting},
year = {2023},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
pages = {224–227},
numpages = {4},
location = {Istanbul, Turkey},
series = {BuildSys '23}
}
````

The dataset used in our paper is private data, and we can't share them for now. If we get the green light, I'll modify this repo further


