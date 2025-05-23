import numpy as np
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.data import get_fnames
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

# Works with .nii.gz exactly the same as .nii
img = nib.load('CT_Philips.nii.gz')  
data = img.get_fdata()

sigma = estimate_sigma(data)

# Simple progress tracking by processing in chunks
print("Starting denoising process...")
total_slices = data.shape[2]

# Process the data and show progress
print("Processing: 0%", end="", flush=True)

denoised = nlmeans(data, sigma=sigma, patch_radius=1, block_radius=2)

print("\rProcessing: 100% - Complete!")

# Save as compressed as well
nib.save(nib.Nifti1Image(denoised, img.affine), 'denoised_mri.nii.gz')

# 3. Display side by side comparison
def display_slices(original, denoised, slice_num=None):
    # If slice_num not provided, use middle slice
    if slice_num is None:
        slice_num = original.shape[2] // 2
    
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    axes[0].imshow(original[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # Display denoised image
    axes[1].imshow(denoised[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[1].set_title('Denoised MRI')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also show the difference map
    plt.figure(figsize=(6, 6))
    difference = np.abs(original[:, :, slice_num] - denoised[:, :, slice_num])
    plt.imshow(difference.T, cmap='hot', origin='lower')
    plt.colorbar(label='Absolute Difference')
    plt.title('Difference Map (Removed Noise)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def interactive_comparison(original, denoised):
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get the middle slice as starting point
    slice_num = original.shape[2] // 2
    
    # Display original and denoised images
    im1 = axes[0].imshow(original[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    im2 = axes[1].imshow(denoised[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[1].set_title('Denoised MRI')
    axes[1].axis('off')
    
    # Display difference map
    difference = np.abs(original[:, :, slice_num] - denoised[:, :, slice_num])
    im3 = axes[2].imshow(difference.T, cmap='hot', origin='lower')
    axes[2].set_title('Difference (Noise Removed)')
    axes[2].axis('off')
    
    # Add slider for navigating through slices
    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, original.shape[2]-1, 
                   valinit=slice_num, valstep=1)
    
    def update(val):
        slice_idx = int(slider.val)
        im1.set_data(original[:, :, slice_idx].T)
        im2.set_data(denoised[:, :, slice_idx].T)
        difference = np.abs(original[:, :, slice_idx] - denoised[:, :, slice_idx])
        im3.set_data(difference.T)
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

def presentation_quality_comparison(original, denoised, slice_num=None):
    if slice_num is None:
        slice_num = original.shape[2] // 2
    
    # Create higher quality visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Original image
    im1 = axes[0].imshow(original[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI', fontsize=14)
    axes[0].axis('off')
    
    # Denoised image
    im2 = axes[1].imshow(denoised[:, :, slice_num].T, cmap='gray', origin='lower')
    axes[1].set_title('Denoised MRI', fontsize=14)
    axes[1].axis('off')
    
    # Difference
    difference = np.abs(original[:, :, slice_num] - denoised[:, :, slice_num])
    im3 = axes[2].imshow(difference.T, cmap='hot', origin='lower')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Noise Magnitude')
    axes[2].set_title('Removed Noise', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle('MRI Denoising with DIPY NLMeans', fontsize=16)
    plt.tight_layout()
    
    # Save high-resolution figure for presentations
    plt.savefig('denoising_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Use this function
interactive_comparison(data, denoised)

# Save the results
nib.save(nib.Nifti1Image(denoised, img.affine), 'denoised_mri.nii.gz')
print("Denoising complete! Saved as 'denoised_mri.nii.gz'")