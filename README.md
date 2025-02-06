# Detection of local prostate cancer recurrence from PET/CT scans using deep learning

This repository contains the accompanying code for the article:

> Korb M, Efetürk H, Jedamzik T, Hartrampf PE, Kosmala A, Serfling SE, Michalski K, Dirk R, Buck AK, Werner RA, Schlötelburg W, and Ankenbrand M. *Detection of local prostate cancer recurrence from PET/CT scans using deep learning*. In preparation

> [!IMPORTANT]
> Large outputs (e.g. model weights) are deposited on Zenodo. Training data can not be shared publically. To understand the structure of the data, these files are included as symbolic links that point outside of the repository.

## Pre-processing

Exported and pseudonomized PET and CT dicom images for each examination were converted to nifti format using [`dcm2niix`](https://github.com/rordenlab/dcm2niix) (Chris Rorden's dcm2niiX version v1.0.20220720  (JP2:OpenJPEG) (JP-LS:CharLS) GCC5.5.0 x86-64 (64-bit Linux)).