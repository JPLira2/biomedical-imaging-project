# Update your denoiser.py file with this:
import numpy as np
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames

# Use a built-in sample instead of your local file
data_path = get_fnames('stanford_t1')  # or 'sherbrooke_3shell' for diffusion MRI
img = nib.load(data_path)
data = img.get_fdata()

# Print some info to verify the file loaded
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Proceed with denoising
sigma = estimate_sigma(data)
print(f"Estimated noise level: {sigma}")

denoised = nlmeans(data, sigma=sigma, patch_radius=1, block_radius=2)

# Save the result
nib.save(nib.Nifti1Image(denoised, img.affine), 'denoised_mri.nii.gz')
print("Denoising complete! Saved as 'denoised_mri.nii.gz'")