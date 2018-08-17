# Project Title

This project implements the speech emotion recognition system on FAU-Aibo corpus.

## Prerequisites

What things you need to install the software.

```
Tensorflow
Keras
Numpy
Matplotlib
Scipy
Python(>= 3.5.5)
```

## Introduction

Our system consists of three parts: static model, dynamic model and ensemble model.

The static model is based on CNN and MLP. The performance measured by unweighted average recall (UA recall) can achieve 46.2%, 46.4% respectively.

The static model is based on RNN. The UA recall can achieve 47.2%.

The ensemble model is based on bagging, maxout-unit and interpolaton. The UA recall implemented by interpolaton can achieve 50.5%.


## Dataset
[FAU-Aibo](https://www5.cs.fau.de/de/mitarbeiter/steidl-stefan/fau-aibo-emotion-corpus/) is a speech emotion database. It is used in [Interspeech 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), including a training set of 9,959 speech chunks and a test set of 8,257 chunks. For the five-category
classification problem, the emotion labels are merged into angry, emphatic, neutral, positive and rest. The data of each
emotion category is summarized in the following Table.


|           | angry | emphatic | neutral | positive | rest | total |
|:---------:|:-----:|:--------:|:-------:|:--------:|:----:|:-----:|
| train set |  881  |   2,093  |  5,590  |    674   |  721 | 9,959 |
|  test set |  611  |   1,508  |  5,377  |    215   |  546 | 8,257 |


## Acoustic features and pre-processing
The acoustic features used in our experiment is the same as those used in the Emotion Challenge extracted by
the [openSMILE](https://audeering.com/technology/opensmile/) toolkit.

For the static model, we use [Cross-speaker Histogram Equalization (CSHE)](http://etd.lib.nsysu.edu.tw/ETD-db/ETD-search/getfile?URN=etd-0730114-173740&filename=etd-0730114-173740.pdf) to reduce the divergence due to speaker while keeping emotional variation.

For the dynamic model, we apply Cepstral mean and variance normalization (CMVN) to eliminate the divergence of signal due to speaker while keeping emotional variation. Each feature dimension of every speaker is normalized to zero mean and unit variance.



## Running the tests
For static model, running the following command:

```
python3 static_model.py
```

For dynamic model, running the following command:

```
python3 dynamic_model.py
```




