# Improving_DeepSpeech_2_by_RNN_Transducer
In this repository, two different ASR Model will be compared in terms of converging speed, wer and cer, one is the DeepSpeech 2 and the second one is the RNN-T based Deep Speech 2.

# Installation
## 1. clone this repository
`git clone https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer.git`  

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

# Model Comparison
## 1. DeepSpeech2
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/DEEPSPEECH2.JPG)


## 2. RNN-Transducer based DeepSpeech 2
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/Transducer.JPG)
