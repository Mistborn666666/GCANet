# GCANet
Pytorch implementation on:GCANet: Enhancing EEG-based Auditory Attention Decoding with Temporal Frequency GCN and Cross Attention Mechanisms

## Introduction
In complex auditory environments, individuals rely on selective auditory attention to focus on a target speaker while suppressing competing sounds, a phenomenon commonly referred to as the cocktail party effect. Auditory attention decoding (AAD) seeks to identify the attended speaker from electroencephalography (EEG) signals. However, most existing approaches overlook the inherent graph-structured nature of EEG. To address this limitation, we propose GCANet, an end-to-end model that integrates a time–frequency graph convolutional network (TFGCN) to capture functional connectivity across brain regions and incorporates a cross-attention mechanism to dynamically enhance interactions between EEG and audio features. Experiments on three publicly available datasets (KUL, DTU, and AVGC) demonstrate that GCANet substantially improves decoding accuracy in both cross-trial and cross-subject evaluations. With a 1-second decision window, GCANet achieves average accuracies of 92.2%, 83.2% and 62.6% in cross-trial settings, and 75.1%, 57.1% and 55.6% in cross-subject settings. Notably, our findings suggest that the alignment between auditory attention and visual cues may introduce gaze-related confounds, which could inadvertently enhance model performance, particularly at shorter decision windows. Furthermore, our analysis indicates that EEG–audio cross-attention highlights consistent involvement of frontal and temporal regions. These findings suggest that the proposed approach can provide useful insights into cross-modal EEG–audio interactions and may inform future research on auditory attention decoding.

## Framework Overview
![GCANet](https://github.com/user-attachments/assets/2b5d86bc-0d02-4b5b-97cd-4ad37c00ed65)
**Fig. 1:** GCANet architecture for AAD, consisting of three components:
1. **TFGCN for EEG** – Multi-channel EEG is processed with a temporal–frequency GCN to capture spatiotemporal-frequency features.
2. **Speech CNN for Audio** – Speech spectrograms are encoded by a CNN to extract complementary audio features.
3. **Cross-Attention Fusion** – EEG and audio features are aligned and fused via cross-attention, and the fused representation is fed into a fully connected layer for AAD classification.

## TF-GCN 
![TF-GCN](https://github.com/user-attachments/assets/1e062733-0851-4161-a7ec-db370d140574)
**Fig. 2.** EEG Temporal-Frequency GCN (TFGCN) module.
1. **Temporal Features** – Multi-scale convolutions extract temporal representations from EEG signals.  
2. **Frequency Features** – DE and PSD maps are combined to construct frequency-domain features.  
3. **Graph Modeling** – TGCN and FGCN capture temporal and frequency dependencies using learnable short- and long-range adjacency matrices, yielding the final EEG embedding.

## Requiements
- Python 3.12
- PyTorch 2.5 (CUDA 12.8)

## Datasets
All datasets used in this paper are publicly available, including [KUL dataset](https://zenodo.org/records/4004271), [DTU dataset](https://zenodo.org/record/1199011) and [AVGC dataset](https://zenodo.org/records/11058711).

## Code Availability
The core source code for reproducing our results are available in the GCANet.ipynb
