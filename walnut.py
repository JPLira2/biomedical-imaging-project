import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

# Works with .nii.gz exactly the same as .nii
img = nib.load('CT_Philips.nii.gz')  
data = img.get_fdata()

# Rest of processing is identical
sigma = estimate_sigma(data)
denoised = nlmeans(data, sigma=sigma)

# Save as compressed as well
nib.save(nib.Nifti1Image(denoised, img.affine), 'denoised_mri.nii.gz')