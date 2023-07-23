# NOT COMPLETED `LSTM Part, Refer to the classifier part` DUE TO MULTIPLE ERROR


# Multimodal Transformer With Learnable Frontend and Self Attention for Emotion Recognition

This repo contains the code for detecting emotion from the conversational dataset IEMOCAP for the implementation of the paper "Multimodal Transformer With Learnable Frontend and Self Attention for Emotion Recognition" submitted to ICASSP 2022. This repository contains the code when Session 5 is conisdered as test and Session 1 as validation.

## Description of the code

- The implementation has three stages, namely, training the unimodal audio and text models, training the Bi-GRU with self-attention and the multimodal transformer
- With the **wav** files for the audio and the **csv** files for text, the first step would be to run **audio_model.py** and the notebook **sentiment_text.ipynb** for audio and text respectively
- The representations from the trained models in the step above are used to create pickle files for the entire dataset
- With these representations, two Bi-GRU models with self-attention (refer to **bigru_audio/text.ipynb**) is trained. The best models for both audio and text are already provided in the **unimodal_models** folder.
- A multimodal transformer is trained on both the modalities of the dataset for the final accuracy results
- Please note that usage of **IEMOCAP** requires permission. Once this is done, we can share the dataset files. For permission please visit [IEMOCAP release](https://sail.usc.edu/iemocap/iemocap_release.htm)

## Tips for running the code of audio_mode.py

## Cloning of `https://github.com/NeelDevenShah/leaf-audio`(forked-repository) or `https://github.com/google-research/leaf-audio`(original repository) and it's installation

- The original version of the leaf-audio is available on https://github.com/google-research/leaf-audio
  From the root directory of the repo, run:

First clone the repository
`git clone https://github.com/NeelDevenShah/leaf-audio`

Second goto that folder by writing
`cd leaf-audio`

Third write for installation
`python setup.py install`

Fourth Restart your system and make following import
`from leaf_audio import frontend`

TADA work done

And leaf-audio is a learnable alternative to audio features such as mel-filterbanks, that can be initialized as an approximation of mel-filterbanks, and then be trained for the task at hand, while using a very small number of parameters.

In-short it is used for making the process of woking with audio easy....

Contact Me at `neeldevenshah@gmail.com`
