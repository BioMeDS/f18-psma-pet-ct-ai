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

### Cropping around prostate (or urinary bladder)

Cropping a 20x20x20 cm³ cube around the centroid of the prostate (if detected) or urinary bladder (otherwise) with the script `code/preprocessing/crop_by_prostate_or_ub.py`.
The cropped files are saved in `data/cropped_nifti`.

### Conversion from BQML to SUV

#### Determine factors for SUV conversion

The converted nifti PET files have values in the BQML unit. In order to convert them to SUV, individual conversion factors have to be determined. This was done by applying `code/preprocessing/bqml_to_suv.py` to the dicom files (containing the relevant header information) to create the factors in `data/suv_factors.tsv`.

#### Convert cropped PET niftis

```bash
while read pid suv
do
	fslmaths data/cropped_nifti/${pid}_pet.nii.gz -mul $suv data/cropped_nifti_suv/${pid}_pet.nii.gz
done <<(tail -n +2 data/suv_factors.tsv)
```

#### Get ranges for scaling

```bash
python code/preprocessing/nii_range.py data/cropped_nifti/*_ct.nii.gz >analysis/cropped_ct_range.tsv
python code/preprocessing/nii_range.py data/cropped_nifti_suv/*_pet.nii.gz >analysis/cropped_pet_suv_range.tsv
```