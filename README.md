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
The diagram shows the architecture of DeepSpeech 2. It consists of 3 ResCNN Layer and 5 Bidirectional GRU Layer and a Connectionist Temporal Classification(CTC) Decoder.  
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/DEEPSPEECH2.JPG)  
The CTC Decoder assumes that **'every output is conditionally independent of the other outputs given the input'** which is not really true in sequence problems such as ASR and NLP. This results that the Char Error Rate(CER) for CTC based ASR System may be guaranteed, though, the Word Error Rate(WER) can not be guaranteed.  

Therefore, in order to make past information available, RNN-Transducer is introduced.
## 2. RNN-T VS CTC
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/CTC_VS_Transducer.JPG)

## 3. RNN-Transducer based DeepSpeech 2
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/Transducer.JPG)
