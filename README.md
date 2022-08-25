# Deep Learning Models for Mispronounce Prediction for Vietnamese Learners of English

By Trang Phung, Mai Tan Ha, Duc-Quang Vu

## Overview
We implement two common deep learning models for Mispronounce Prediction for Vietnamese Learners of English. 
Two models used in our approach include CNN and LSTM.
Both models have been conducted on the L2-ARCTIC dataset. 
The experiment results have shown that both models gave a positive performance. 
Besides of having the memory mechanism, the LSTM model outperforms 6.3% compared to the CNN model. 
In the future, we will continue to delve into the exploitation of more advanced datasets and models (such as the Transformer) to solve the problem with higher results.


<p align="center">
  <img width="600" alt="fig_method" src="demo.png">
</p>


## Running the code

### Requirements
- Python3
- Tensorflow (>=2.3.0)
- numpy 
- Pillow
- librosa
- pystoi
- pesq
- mir_eval
- 
### Prepare data
We use the L2-ARCTIC dataset in our experiment. 
Therefore, to run this code, you need to download the L2-ARCTIC dataset from https://psi.engr.tamu.edu/l2-arctic-corpus/

We have 4 folders with Vietnamese speakers including "HQTV", "PNV", "THV" and "TLV". 
These folders should be put in the folder "dataset". 
Note that all files (including annotation and wav files) in 4 folders above should be keep original version. 
We don't change anything in these files.


### Training

In this code, you can reproduce the experimental results of the Mispronounce Prediction for Vietnamese Learners of English in the submitted paper as follows:


- Training with SML
~~~
python train_model.py --gpu=0 --model=cnn --lr=0.01 --drop_rate=0.5 --reg_factor=5e-4 --epochs=100 --batch_size=64 
~~~

In which, 

`--gpu=0` denotes which GPU we use to train the network

`--model=cnn` is the which network we use to address the task (`cnn` or `lstm`)

`--lr=0.01` means the learning rate init

`--drop_rate=0.5` is the drop rate of the dropout layer

`--reg_factor=5e-4` is the regularization value of the l2 norm

`--epochs=100` is the total epochs for training

`--batch_size=64` is the batch size



