# Augmenting Multilingual Language Models with Human Reading Behavior - Master's Thesis - Sven Spa

This repository provides the code used for my Master's Thesis (NR. 406).

[Linear Classifier](https://github.com/svenspa/Gaze_Multilingual_LMs/blob/master/linear_class.py) is the script used for the Language identification task using eye-tracking data.

[MAN-MoE model](https://github.com/svenspa/Gaze_Multilingual_LMs/tree/master/MAN_MoE_model) is the folder containing all the code used to run the experiments with the MAN-MoE model. The non-augmented/baseline model is developed by [Microsoft](https://github.com/svenspa/Multilingual-Model-Transfer).

[Embedding model](https://github.com/svenspa/Gaze_Multilingual_LMs/tree/master/embedding_model) is the folder containing all the code to generate the multilingual embedding space. This is based on an existing model named [Unsupervised Multilingual Word Embeddings](https://github.com/ccsasuke/umwe) developed by Xilun Chen and Claire Cardie. Their model is based on [MUSE](https://github.com/facebookresearch/MUSE) by Meta Research.

The eye-tracking data is the L1 release from [MECO](https://osf.io/3527a/).

All the models are trained using multiple NVIDIA TITAN Xp GPUs. 

## Requirements (besides the standard packages)

- Python 3.6.13
- PyTorch 1.8.0
- Faiss 1.7.2
- eli5 0.13.0

## Data
- [CoNLL 2002](https://www.clips.uantwerpen.be/conll2002/ner/)
- [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [Wikiann](https://huggingface.co/datasets/wikiann)
- [MECO](https://osf.io/3527a/)

## Run Experiment
- [MAN-MoE model](https://github.com/svenspa/Gaze_Multilingual_LMs/tree/master/MAN_MoE_model) provides bash scripts with you can run with arguments.
- [Embedding model](https://github.com/svenspa/Gaze_Multilingual_LMs/tree/master/embedding_model) you can run via `unsupervised.py` with arguments.

## Contact
[LinkedIn](www.linkedin.com/in/sven-spa-1b4269b3)
