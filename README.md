#  SARL: SUBJECT-ADAPTIVE RELIABILITY LEARNING FOR MULTIMODAL SPEECH-BASED COGNITIVE IMPAIRMENT DETECTION

This repository contains the official implementation of **SARL**, a multimodal framework for cognitive impairment detection using speech and text representations extracted from pretrained foundation models.

**Lili Zheng<sup>1</sup>, Hong Liang<sup>1</sup>, Sichen Li<sup>3</sup>, Yue Hua<sup>4</sup>, Chen Jason Zhang<sup>2</sup>, Qi Shao<sup>1</sup>, Baoru Huang<sup>5</sup>, Shan Cong<sup>1,6,†</sup>, Xiaohui Yao<sup>1,†</sup>, Haoran Luo<sup>2,†</sup>**

<sup>1</sup> College of Intelligent Systems Science and Engineering, Harbin Engineering University, Harbin, China  
<sup>2</sup> The Hong Kong Polytechnic University, Hong Kong, China  
<sup>3</sup> College of Computing & Data Science, Nanyang Technological University, Singapore  
<sup>4</sup> School of Traditional Chinese Medicine, Southern Medical University, Guangzhou, China  
<sup>5</sup> School of Computer Science and Informatics, University of Liverpool, Liverpool, United Kingdom  
<sup>6</sup> Stem Cell and Regenerative Biology, Genome Institute of Singapore, A*STAR, Singapore

---

## Code and Data Availability

The datasets used in this study are publicly available from their respective data providers under specific access requirements and data-use agreements.

- **ADReSS dataset**  
  Available through the DementiaBank platform:  
  https://talkbank.org/dementia/ADReSS-2020/

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

This work was developed at Harbin Engineering University.
We sincerely thank all dataset providers, participants, and collaborators for making this research possible.

---

## Citation

Citation information will be provided upon publication of the corresponding manuscript.
