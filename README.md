# Detection of local prostate cancer recurrence from PET/CT scans using deep learning

This repository contains the accompanying code for the article:

> Korb M, Efetürk H, Jedamzik T, Hartrampf PE, Kosmala A, Serfling SE, Michalski K, Dirk R, Buck AK, Werner RA, Schlötelburg W, and Ankenbrand M. *Detection of local prostate cancer recurrence from PET/CT scans using deep learning*. In preparation

> [!IMPORTANT]
> Large outputs (e.g. model weights) are deposited on Zenodo ([10.5281/zenodo.14944880](https://doi.org/10.5281/zenodo.14944880)). Training data can not be shared publically. To understand the structure of the data, these files are included as symbolic links that point outside of the repository.

## Pre-processing

### Nifti conversion
Exported and pseudonomized PET and CT dicom images for each examination were converted to nifti format using [`dcm2niix`](https://github.com/rordenlab/dcm2niix) (Chris Rorden's dcm2niiX version v1.0.20220720  (JP2:OpenJPEG) (JP-LS:CharLS) GCC5.5.0 x86-64 (64-bit Linux)). Those niftis are saved in `data/nifti` (train and validation set) and `data/nifti_ts2024` (test set).

### Prostate segmentation
The prostate and urinary bladder were segmented with [`TotalSegmentator`](https://github.com/wasserth/TotalSegmentator) (version 2.1.0) in all ct images.

```bash
for i in data/nifti*/*_ct.nii.gz
do
	TotalSegmentator -i $i -o analysis/totalsegmentator2/$(basename $i _ct.nii.gz) -rs prostate urinary_bladder
done
```