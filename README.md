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
<div align=center>![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/CTC_VS_Transducer.JPG)</div>  

As the graph shows that, instead of merely focus on acoustic features/spectrogram x<sub>1</sub>-x<sub>T</sub>, the RNN-T also regard y<sub>u-1</sub> as its input. Moreover, RNN-T has one more **Prediction Network**, namely, Decoder compared with CTC, which learn the relationship between y<sub>u</sub> and y<sub>u-1</sub>

## 3. RNN-Transducer based DeepSpeech 2
![Image](https://github.com/MorrisXu-Driving/Improving_DeepSpeech_2_by_RNN_Transducer/blob/master/readme_img/Transducer.JPG)  
This diagram shows the model architecture in this repository. Please read the code with this graph to have better understanding.

# Result Comparison


# Reference
* [IMPROVING RNN TRANSDUCER MODELING FOR END-TO-END SPEECH RECOGNITION](https://arxiv.org/pdf/1909.12415.pdf)
* [A Comparison of Sequence-to-Sequence Models for Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF)
* [warp-transducer with CUDA by @HawkAaron](https://github.com/HawkAaron/warp-transducer)
* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf)

