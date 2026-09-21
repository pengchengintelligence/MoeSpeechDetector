# Code-MOE

This repository contains the official implementation of **Code-MOE**, a multimodal framework for cognitive impairment detection using speech and text representations extracted from pretrained foundation models.

---

## Code and Data Availability

The datasets used in this study are publicly available from their respective data providers under specific access requirements and data-use agreements.

- **ADReSS dataset**  
  Available through the DementiaBank platform:  
  https://talkbank.org/dementia/ADReSS-2020/

- **ADReSSo dataset**  
  Available through the DementiaBank platform:  
  https://talkbank.org/dementia/ADReSSo-2021/

- **NCMMSC2021 Alzheimer's Disease Recognition dataset**  
  Available through the official dataset page:  
  https://web.ee.tsinghua.edu.cn/satlab/en/gxsj/7552/content/1011.htm

- **TAUKADIAL dataset**  
  Available through DementiaBank:  
  https://talkbank.org/dementia/TAUKADIAL/index.html

Access to DementiaBank datasets requires membership and approval for research use.

---

## Requirements

The implementation is based on Python and PyTorch.

- Python >= 3.8
- PyTorch
- Transformers
- scikit-learn
- pandas
- NumPy
- librosa
- soundfile
- tqdm

---

## Data Preparation

The datasets should be downloaded from their corresponding official sources and prepared according to the preprocessing pipeline described in the paper.

The main scripts are provided for:

- speech feature extraction: `script/extract_speech_layered_features.py`
- text feature extraction: `script/extract_text_layered_features.py`
- cached feature construction: `script/build_dca_moe_cached_features.py`
- model training and evaluation: `script/train_cv5_layered_dirichlet_tcp_fixed_s_full_seoc.py`

Before feature extraction, please modify the local paths of the pretrained speech and language models in the corresponding feature extraction scripts.

---

## Disclaimer

This repository is intended for **academic research purposes only**.

The proposed method has not been approved as a clinical diagnostic tool and should not be used for medical decision-making.

---

## Acknowledgments

We sincerely thank all dataset providers, participants, and collaborators for making this research possible.

---

## Citation

Citation information will be provided upon publication of the corresponding manuscript.
