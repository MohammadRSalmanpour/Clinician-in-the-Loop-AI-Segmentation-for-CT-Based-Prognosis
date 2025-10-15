"""
Complete Preprocessing Pipeline for Lung CT Segmentation
Author: Sonya Falahati
Description: End-to-end preprocessing pipeline for lung CT images including:
- Resampling to uniform spacing
- Hole filling in segmentation masks
- Largest tumor selection
- ROI cropping with margin

Usage:
    python preprocessing.py
"""

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import label

class LungCTPreprocessor:
    def __init__(self, base_path="./train/"):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            base_path (str): Base directory for input/output data
        """
        self.base_path = base_path
        self.config = {
            'new_spacing': (1.0, 1.0, 1.0),
            'min_voxel_count': 5,
            'min_tumor_volume_mm3': 10.0,
            'crop_margin': 5
        }
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = [
            "./step1/Resampled_CT",
            "./step1/Resampled_seg", 
            "./step2/Filled_seg",
            "./step3/Largest_tumors/seg",
            "./step4/ROI_cropped/Cropped_CT",
            "./step4/ROI_cropped/Cropped_seg"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.base_path, directory), exist_ok=True)
    
    def run_complete_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("🚀 Starting Lung CT Preprocessing Pipeline...")
        
        # Step 1: Resampling
        print("\n" + "="*60)
        print("STEP 1: Resampling Images")
        print("="*60)
        self._resample_images()
        
        # Step 2: Fill Holes
        print("\n" + "="*60)
        print("STEP 2: Filling Mask Holes") 
        print("="*60)
        self._fill_mask_holes()
        
        # Step 3: Largest Tumor Selection
        print("\n" + "="*60)
        print("STEP 3: Selecting Largest Tumor")
        print("="*60)
        self._select_largest_tumor()
        
        # Step 4: ROI Cropping
        print("\n" + "="*60)
        print("STEP 4: ROI Cropping")
        print("="*60)
        self._crop_roi()
        
        print("\n✅ Preprocessing pipeline completed successfully!")
    
    def _resample_images(self):
        """Step 1: Resample CT and mask images to uniform spacing"""
        input_ct_dir = os.path.join(self.base_path, "CT")
        input_mask_dir = os.path.join(self.base_path, "Seg")
        output_ct_dir = os.path.join(self.base_path, "./step1/Resampled_CT")
        output_mask_dir = os.path.join(self.base_path, "./step1/Resampled_seg")
        output_excel = os.path.join(self.base_path, "./step1/Resampled_data.xlsx")
        
        # Gather files
        ct_files = sorted([f for f in os.listdir(input_ct_dir) if f.endswith((".nii", ".nii.gz"))])
        mask_files = sorted([f for f in os.listdir(input_mask_dir) if f.endswith((".nii", ".nii.gz"))])
        
        assert len(ct_files) == len(mask_files), "Mismatch between CT and mask files!"
        
        combined_info = []
        
        for idx, filename in enumerate(ct_files, start=1):
            print(f"📁 Processing {idx}/{len(ct_files)}: {filename}")
            
            # Load images
            ct_path = os.path.join(input_ct_dir, filename)
            mask_path = os.path.join(input_mask_dir, filename)
            
            ct_image = sitk.ReadImage(ct_path)
            mask_image = sitk.ReadImage(mask_path)
            
            orig_size = ct_image.GetSize()
            orig_spacing = ct_image.GetSpacing()
            
            # Calculate new size
            new_size = [
                int(round(orig_size[i] * (orig_spacing[i] / self.config['new_spacing'][i])))
                for i in range(3)
            ]
            
            # Resample CT (Linear interpolation)
            ct_resampler = sitk.ResampleImageFilter()
            ct_resampler.SetOutputSpacing(self.config['new_spacing'])
            ct_resampler.SetSize(new_size)
            ct_resampler.SetOutputDirection(ct_image.GetDirection())
            ct_resampler.SetOutputOrigin(ct_image.GetOrigin())
            ct_resampler.SetInterpolator(sitk.sitkLinear)
            ct_resampled = ct_resampler.Execute(ct_image)
            
            # Resample Mask (Nearest Neighbor)
            mask_resampler = sitk.ResampleImageFilter()
            mask_resampler.SetOutputSpacing(self.config['new_spacing'])
            mask_resampler.SetSize(new_size)
            mask_resampler.SetOutputDirection(mask_image.GetDirection())
            mask_resampler.SetOutputOrigin(mask_image.GetOrigin())
            mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resampled = mask_resampler.Execute(mask_image)
            
            # Save resampled images
            ct_output_path = os.path.join(output_ct_dir, filename)
            mask_output_path = os.path.join(output_mask_dir, filename)
            sitk.WriteImage(ct_resampled, ct_output_path)
            sitk.WriteImage(mask_resampled, mask_output_path)
            
            # Collect metadata
            case_info = {
                "Image/Mask Name": filename,
                "Original Size (x, y, z)": str(orig_size),
                "Original Spacing (x, y, z)": str(orig_spacing),
                "Resample Size (x, y, z)": str(ct_resampled.GetSize()),
                "New Spacing (x, y, z)": str(ct_resampled.GetSpacing()),
            }
            combined_info.append(case_info)
        
        # Save metadata
        df = pd.DataFrame(combined_info)
        df.to_excel(output_excel, index=False)
        print(f"📊 Resampling metadata saved to: {output_excel}")
    
    def _fill_mask_holes(self):
        """Step 2: Fill holes in segmentation masks"""
        input_mask_dir = os.path.join(self.base_path, "./step1/Resampled_seg")
        output_mask_dir = os.path.join(self.base_path, "./step2/Filled_seg")
        
        mask_files = sorted([f for f in os.listdir(input_mask_dir) if f.endswith((".nii", ".nii.gz"))])
        
        for idx, filename in enumerate(mask_files, start=1):
            print(f"🔄 Filling holes {idx}/{len(mask_files)}: {filename}")
            
            mask_path = os.path.join(input_mask_dir, filename)
            mask_image = sitk.ReadImage(mask_path)
            
            # Convert to binary and fill holes
            mask_array = sitk.GetArrayFromImage(mask_image)
            binary_mask_array = (mask_array > 0).astype(np.uint8)
            binary_mask_image = sitk.GetImageFromArray(binary_mask_array)
            binary_mask_image.CopyInformation(mask_image)
            
            hole_filler = sitk.BinaryFillholeImageFilter()
            filled_mask_image = hole_filler.Execute(binary_mask_image)
            
            # Save filled mask
            filled_mask_path = os.path.join(output_mask_dir, filename)
            sitk.WriteImage(filled_mask_image, filled_mask_path)
    
    def _select_largest_tumor(self):
        """Step 3: Select largest connected tumor component"""
        input_mask_dir = os.path.join(self.base_path, "./step2/Filled_seg")
        output_mask_dir = os.path.join(self.base_path, "./step3/Largest_tumors/seg")
        output_excel = os.path.join(self.base_path, "./step3/tumor_analysis.xlsx")
        
        mask_files = sorted([f for f in os.listdir(input_mask_dir) if f.endswith((".nii", ".nii.gz"))])
        tumor_analysis_data = []
        
        for idx, filename in enumerate(mask_files, start=1):
            print(f"🎯 Selecting largest tumor {idx}/{len(mask_files)}: {filename}")
            
            mask_path = os.path.join(input_mask_dir, filename)
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)
            
            # Identify connected components
            labeled_array, num_features = label(mask_array)
            voxel_volume = np.prod(mask_image.GetSpacing())
            
            valid_tumor_volumes = []
            valid_tumor_labels = []
            
            # Filter tumors based on size criteria
            for tumor_label in range(1, num_features + 1):
                tumor_mask = labeled_array == tumor_label
                voxel_count = np.sum(tumor_mask)
                tumor_volume_mm3 = voxel_count * voxel_volume
                
                if (voxel_count >= self.config['min_voxel_count'] and 
                    tumor_volume_mm3 >= self.config['min_tumor_volume_mm3']):
                    valid_tumor_volumes.append(tumor_volume_mm3)
                    valid_tumor_labels.append(tumor_label)
            
            # Keep only the largest tumor
            new_mask_array = np.zeros_like(labeled_array)
            largest_tumor_volume = 0.0
            
            if valid_tumor_volumes:
                largest_tumor_idx = np.argmax(valid_tumor_volumes)
                largest_tumor_label = valid_tumor_labels[largest_tumor_idx]
                largest_tumor_volume = valid_tumor_volumes[largest_tumor_idx]
                new_mask_array[labeled_array == largest_tumor_label] = 1
            
            # Save updated mask
            new_mask_image = sitk.GetImageFromArray(new_mask_array.astype(np.uint8))
            new_mask_image.CopyInformation(mask_image)
            new_mask_path = os.path.join(output_mask_dir, filename)
            sitk.WriteImage(new_mask_image, new_mask_path)
            
            # Collect analysis data
            tumor_analysis_data.append({
                "File Name": filename,
                "Initial Tumor Count": num_features,
                "Valid Tumor Count": len(valid_tumor_volumes),
                "Largest Tumor Volume (mm³)": largest_tumor_volume
            })
        
        # Save tumor analysis
        tumor_df = pd.DataFrame(tumor_analysis_data)
        tumor_df.to_excel(output_excel, index=False)
        print(f"📊 Tumor analysis saved to: {output_excel}")
    
    def _crop_roi(self):
        """Step 4: Crop region of interest around tumor"""
        input_ct_dir = os.path.join(self.base_path, "./step1/Resampled_CT")
        input_mask_dir = os.path.join(self.base_path, "./step3/Largest_tumors/seg")
        output_ct_dir = os.path.join(self.base_path, "./step4/ROI_cropped/Cropped_CT")
        output_mask_dir = os.path.join(self.base_path, "./step4/ROI_cropped/Cropped_seg")
        output_excel = os.path.join(self.base_path, "./step4/roi_info.xlsx")
        
        ct_files = sorted([f for f in os.listdir(input_ct_dir) if f.endswith((".nii", ".nii.gz"))])
        roi_data = []
        
        for filename in ct_files:
            print(f"✂️ Cropping ROI: {filename}")
            
            try:
                ct_path = os.path.join(input_ct_dir, filename)
                mask_path = os.path.join(input_mask_dir, filename)
                
                ct_image = sitk.ReadImage(ct_path)
                mask_image = sitk.ReadImage(mask_path)
                
                ct_data = sitk.GetArrayFromImage(ct_image)
                mask_data = sitk.GetArrayFromImage(mask_image)
                
                # Get bounding box
                coords = np.argwhere(mask_data > 0)
                if len(coords) == 0:
                    print(f"⚠️ No tumor found in {filename}, skipping...")
                    continue
                    
                x_min, y_min, z_min = coords.min(axis=0)
                x_max, y_max, z_max = coords.max(axis=0)
                bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
                
                # Apply margin
                x_min = max(0, x_min - self.config['crop_margin'])
                x_max = min(ct_data.shape[0] - 1, x_max + self.config['crop_margin'])
                y_min = max(0, y_min - self.config['crop_margin'])
                y_max = min(ct_data.shape[1] - 1, y_max + self.config['crop_margin'])
                z_min = max(0, z_min - self.config['crop_margin'])
                z_max = min(ct_data.shape[2] - 1, z_max + self.config['crop_margin'])
                
                # Crop images
                cropped_ct = ct_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
                cropped_mask = mask_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
                
                # Save cropped images
                cropped_ct_image = sitk.GetImageFromArray(cropped_ct)
                cropped_ct_image.CopyInformation(ct_image)
                sitk.WriteImage(cropped_ct_image, os.path.join(output_ct_dir, filename))
                
                cropped_mask_image = sitk.GetImageFromArray(cropped_mask)
                cropped_mask_image.CopyInformation(mask_image)
                sitk.WriteImage(cropped_mask_image, os.path.join(output_mask_dir, filename))
                
                # Collect ROI data
                roi_data.append({
                    "File Name": filename,
                    "Original Size (x, y, z)": ct_data.shape,
                    "Cropped ROI Size (with margin)": cropped_ct.shape,
                    "Margin Applied (voxels)": self.config['crop_margin'],
                    "Crop Start Coordinates (x, y, z)": (x_min, y_min, z_min)
                })
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        # Save ROI information
        roi_df = pd.DataFrame(roi_data)
        roi_df.to_excel(output_excel, index=False)
        print(f"📊 ROI information saved to: {output_excel}")

def main():
    """Main function to run the preprocessing pipeline"""
    preprocessor = LungCTPreprocessor(base_path="./train/")
    preprocessor.run_complete_pipeline()

if __name__ == "__main__":
    main()