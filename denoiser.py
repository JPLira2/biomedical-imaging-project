import numpy as np
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames


data_path = get_fnames('stanford_t1')  # or any mri we want
img = nib.load(data_path)
data = img.get_fdata()

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

sigma = estimate_sigma(data)
print(f"Estimated noise level: {sigma}")

denoised = nlmeans(data, sigma=sigma, patch_radius=1, block_radius=2)

nib.save(nib.Nifti1Image(denoised, img.affine), 'denoised_mri.nii.gz')
print("Denoising complete! Saved as 'denoised_mri.nii.gz'")