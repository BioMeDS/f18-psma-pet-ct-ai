import nibabel
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_niftis_by_id(i, test_set=False):
    ts = "_ts2024" if test_set else ""
    pet = nibabel.load(f"/data/f18-psma-pet-ct-ml/data/nifti{ts}/{i}_pet.nii.gz")
    ct = nibabel.load(f"/data/f18-psma-pet-ct-ml/data/nifti{ts}/{i}_ct.nii.gz")
    seg_pro = nibabel.load(f"analysis/totalsegmentator2/{i}/prostate.nii.gz")
    seg_ub = nibabel.load(f"analysis/totalsegmentator2/{i}/urinary_bladder.nii.gz")
    return pet, ct, seg_pro, seg_ub

def get_center(seg):
    return np.array([np.mean(x) for x in np.where(seg.dataobj)])

def from_ct_to_patient(coords, ct):
    return np.matmul(ct.affine[:3,:3],coords) + ct.affine[:3,3]

def round_and_clip(coords, full_shape):
	return np.clip(np.round(coords), [0,0,0], full_shape).astype(np.int16)

def from_patient_to_pix(coords, pix):
    coords = np.matmul(np.linalg.inv(pix.affine[:3,:3]), coords - pix.affine[:3,3])
    return round_and_clip(coords, pix.shape)

def crop_pet_ct_by_pro_or_ub(pet, ct, seg_pro, seg_ub):
    center = get_center(seg_pro) if seg_pro.get_fdata().sum() > 0 else get_center(seg_ub)
    start_ct = from_patient_to_pix(from_ct_to_patient(center, ct)-100,ct)
    end_ct = from_patient_to_pix(from_ct_to_patient(center, ct)+100,ct)
    start_pet = from_patient_to_pix(from_ct_to_patient(center, ct)-100,pet)
    end_pet = from_patient_to_pix(from_ct_to_patient(center, ct)+100,pet)
    ct_cropped = np.array(ct.dataobj)[end_ct[0]:start_ct[0],start_ct[1]:end_ct[1],start_ct[2]:end_ct[2]]
    pet_cropped = np.array(pet.dataobj)[end_pet[0]:start_pet[0],start_pet[1]:end_pet[1],start_pet[2]:end_pet[2]]
    return pet_cropped, ct_cropped, start_pet, start_ct, end_pet, end_ct

ids = pd.read_csv("data/labels_ts2024.tsv", sep="\t").pseudo_id

for i in tqdm(ids):
    pet,ct,seg_pro,seg_ub = get_niftis_by_id(i, test_set=True)
    pet_cropped, ct_cropped, start_pet, start_ct, end_pet, end_ct = crop_pet_ct_by_pro_or_ub(pet, ct, seg_pro, seg_ub)
    start_ct_int = np.round(start_ct).astype(np.int16)
    end_ct_int = np.round(end_ct).astype(np.int16)
    start_pet_int = np.round(start_pet).astype(np.int16)
    end_pet_int = np.round(end_pet).astype(np.int16)
    cropped_pet_affine = pet.affine.copy()
    cropped_pet_affine[:3,3] = from_ct_to_patient(start_pet_int, pet)
    cropped_ct_affine = ct.affine.copy()
    cropped_ct_affine[:3,3] = from_ct_to_patient(start_ct_int, ct)
    cropped_pet_nii = nibabel.Nifti1Image(pet_cropped, cropped_pet_affine, pet.header)
    nibabel.save(cropped_pet_nii, f"data/cropped_nifti/{i}_pet.nii.gz")
    cropped_ct_nii = nibabel.Nifti1Image(ct_cropped, cropped_ct_affine, ct.header)
    nibabel.save(cropped_ct_nii, f"data/cropped_nifti/{i}_ct.nii.gz")