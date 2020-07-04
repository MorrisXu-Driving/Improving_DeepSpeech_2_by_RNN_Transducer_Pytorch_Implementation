# Improving_DeepSpeech_2_by_RNN_Transducer
In this repository, two different ASR Model will be compared in terms of converging speed, wer and cer, one is the DeepSpeech 2 and the second one is the RNN-T based Deep Speech 2.

# Installation
## 1. clone this repository
`git clone https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer.git`  
Then two .py files are under your project directory. The `Deep Speech 2.py` is an implementation of Deep Speech 2, an ASR Model based on DL, lauched in 2015.
The `DeepSpeech 2 with LSTM Transducer.py` is an implementation of improved Deep Speech 2 with RNN-Transudcer in Pytorch.

## 2. RNNT Loss Library Installation
This library is developed by @HawkAaron. It is a RNNT Loss function with CUDA which accelerates the training process.    
```
cd Improving_DeepSpeech_2_by_RNN_Transducer
git clone https://github.com/HawkAaron/warp-transducer.git
cd warp-transducer
```
The following detailed installation can be found at https://github.com/HawkAaron/warp-transducer/

## 3. Other compulsory libraries
Before running the DeepSpeech 2 with LSTM Transducer.py, please make sure that `comet_ml`,`torch`,`torchaudio` are installed. No version restriction.
If not, please run:  
`pip install torchaudio torch comet-ml`

## 4. GPU Requirement
I ran this script on Google Cloud GPU VM with the following detailed configurations. Please compare the information below with your server accordingly before running.  
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/GPU.JPG)

# Model Comparison
## 1. DeepSpeech2
The diagram shows the architecture of DeepSpeech 2. It consists of 3 ResCNN Layer and 5 Bidirectional GRU Layer and a Connectionist Temporal Classification(CTC) Decoder.  
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/DEEPSPEECH2.JPG)  
The CTC Decoder assumes that **'every output is conditionally independent of the other outputs given the input'** which is not really true in sequence problems such as ASR and NLP. This results that the Char Error Rate(CER) for CTC based ASR System may be guaranteed, though, the Word Error Rate(WER) can not be guaranteed.  

Therefore, in order to make past information available, RNN-Transducer is introduced.
## 2. RNN-T VS CTC
<div align=center><img src="https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/CTC_VS_Transducer.JPG"></div>


As the graph shows that, instead of merely focus on acoustic features/spectrogram x<sub>1</sub>-x<sub>T</sub>, the RNN-T also regard y<sub>u-1</sub> as its input. Moreover, RNN-T has one more **Prediction Network**, namely, Decoder compared with CTC, which learn the relationship between y<sub>u</sub> and y<sub>u-1</sub>

## 3. RNN-Transducer based DeepSpeech 2
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/Transducer.JPG)  
This diagram shows the model architecture in this repository. Please read the code with this graph to have better understanding. The RNN-Transducer Structure consider not only  the input acoustic features but also the output labels from *t-1*.

# Result Comparison
## CER
| Epoch | DeepSpeech2   | RNN-Transduced DeepSpeech2 |
| :------:| :-------------: | :-------------: |
| 1| 0.432  | 0.407  |
| 2| 0.327  | 0.308  |
| 3| 0.274  | 0.312  |
| 4| 0.223  | 0.294  |
| 5| 0.207  | 0.257  |
| 6| 0.192  | 0.201  |
| 7| 0.188  | 0.184  |
| 8| 0.186  | 0.176  |
| 9| 0.184  | 0.172  |
| 10| 0.185  | 0.171  |

## WER
| Epoch | DeepSpeech2   | RNN-Transduced DeepSpeech2 |
| :------:| :-------------: | :-------------: |
| 1| 0.992  |  0.650  |
| 2| 0.832  |  0.641  |
| 3| 0.718  | 0.592  |
| 4| 0.634  | 0.551  |
| 5| 0.587  | 0.538  |
| 6| 0.549  | 0.514  |
| 7| 0.540  | 0.497  |
| 8| 0.534  | 0.489  |
| 9| 0.529  | 0.484  |
| 10| 0.527 | 0.479  |

# Reference
* [IMPROVING RNN TRANSDUCER MODELING FOR END-TO-END SPEECH RECOGNITION](https://arxiv.org/pdf/1909.12415.pdf)
* [A Comparison of Sequence-to-Sequence Models for Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF)
* [warp-transducer with CUDA by @HawkAaron](https://github.com/HawkAaron/warp-transducer)
* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf)

