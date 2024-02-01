#Copyright 2023 Diagnostic Image Analysis Group, Radboud
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


import os
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# imports required for running nnUNet algorithm
import subprocess
from pathlib import Path

# imports required for my algorithm
import SimpleITK as sitk
from data_utils import resample_img, GetROIfromDownsampledSegmentation, FPreductionPancreasMaskEnsamble


class PDACDetectionContainer(SegmentationAlgorithm):
    def __init__(self, output_raw_heatmap: bool = False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # configure algorithm options
        self.output_raw_heatmap = output_raw_heatmap

        # input / output paths for nnUNet
        self.nnunet_input_dir_lowres = Path("/opt/algorithm/nnunet/input_lowres") 
        self.nnunet_input_dir_fullres = Path("/opt/algorithm/nnunet/input_fullres")
        self.nnunet_output_dir_lowres = Path("/opt/algorithm/nnunet/output_lowres")
        self.nnunet_output_dir_fullres = Path("/opt/algorithm/nnunet/output_fullwres")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/results")

        # input / output paths
        self.ct_ip_dir         = Path("/input/images/")
        self.output_dir        = Path("/output/images/")
        self.output_dir_tlm    = self.output_dir / "pancreatic-tumor-likelihood-map"
        self.output_dir_seg    = self.output_dir / "pancreas-anatomy-and-vessel-segmentation"
        self.heatmap           = self.output_dir_tlm / "heatmap.mha"
        self.heatmap_raw       = self.output_dir_tlm / "heatmap_raw.mha"
        self.segmentation      = self.output_dir_seg / "segmentation.mha"

        # ensure required folders exist
        self.nnunet_input_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_input_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_tlm.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg.mkdir(exist_ok=True, parents=True)

        # try to find the input CT image
        try:
            self.ct_image = next(self.ct_ip_dir.glob("*.mha"))
            print(f"Found input CT image: {self.ct_image}")
        except StopIteration:
            self.ct_image = None
            print(f"Warning: no input CT image found")

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load CT scan and Generate Heatmap for Pancreas Cancer  
        """
        itk_img    = sitk.ReadImage(str(self.ct_image), sitk.sitkFloat32)
        image_np   = sitk.GetArrayFromImage(itk_img)

        #Get low resolution pancreas segmentation 
        #dowsample image to 256x256
        original_spacing = itk_img.GetSpacing()
        original_size    = itk_img.GetSize()
        initial_spacing  = np.array(original_spacing)
        print('original_size: ', original_size)
        if (original_size[0]>256):
            scale = original_size[0]/256
            output_spacing = scale*initial_spacing
            resampled_image = resample_img(itk_img, output_spacing)

        print('resampled_image_size: ', resampled_image.GetSize())

        # save resampled image
        sitk.WriteImage(resampled_image, str(self.nnunet_input_dir_lowres / "scan_0000.nii.gz"))
   
        #predict pancreas mask using nnUnet
        self.predict(
            input_dir=self.nnunet_input_dir_lowres,
            output_dir=self.nnunet_output_dir_lowres,
            task="Task105_PancreasDownsampledres256",
            trainer="nnUNetTrainerV2"
        )
        mask_pred_path = str(self.nnunet_output_dir_lowres / "scan.nii.gz")
        mask_low_res = sitk.ReadImage(mask_pred_path)

        cropped_image, coordinates = GetROIfromDownsampledSegmentation(itk_img, resampled_image, mask_low_res, 80,50,10)
        # save cropped image
        sitk.WriteImage(cropped_image, str(self.nnunet_input_dir_fullres / "scan_0000.nii.gz"))

        # Predict using nnUNet ensemble, averaging multiple restarts
        # also need to store the nii.gz predictions for the post-processing

        self.predict(
            input_dir=self.nnunet_input_dir_fullres,
            output_dir=self.nnunet_output_dir_fullres,
            task="Task103_AllStructures",
            trainer="nnUNetTrainerV2_Loss_CE_checkpoints"
        )
        pred_path_np = str(self.nnunet_output_dir_fullres / "scan.npz")
        pred_path_nii = str(self.nnunet_output_dir_fullres / "scan.nii.gz")

        pred_1 = np.load(pred_path_np)['softmax'][1].astype(np.float32)
        pred_1_nii = sitk.ReadImage(pred_path_nii)
        self.predict(
            input_dir=self.nnunet_input_dir_fullres,
            output_dir=self.nnunet_output_dir_fullres,
            task="Task103_AllStructures",
            trainer="nnUNetTrainerV2_Loss_CE_checkpoints2"
        )
        pred_2 = np.load(pred_path_np)['softmax'][1].astype(np.float32)
        pred_2_nii = sitk.ReadImage(pred_path_nii)
        pred_2_np  = sitk.GetArrayFromImage(pred_2_nii).astype(np.uint8)
        #Remove tumour and tumour thrombosis segmentation and reorder
        pred_2_np[pred_2_np==1]=0
        pred_2_np[pred_2_np==8]=0
        pred_2_np[pred_2_np==2]=1
        pred_2_np[pred_2_np==3]=2
        pred_2_np[pred_2_np==4]=3
        pred_2_np[pred_2_np==5]=4
        pred_2_np[pred_2_np==6]=5
        pred_2_np[pred_2_np==7]=6
        pred_2_np[pred_2_np==9]=7

        pred_ensemble = (pred_1 + pred_2)/2

        softmax_tumor_masked = FPreductionPancreasMaskEnsamble(pred_1_nii, pred_2_nii, pred_ensemble, True)

        pm_image = np.zeros(image_np.shape, dtype=np.float32)
        segmentation_np = np.zeros(image_np.shape, dtype=np.uint8)

        pm_image[coordinates['z_start']:coordinates['z_finish'],
                 coordinates['y_start']:coordinates['y_finish'],
                 coordinates['x_start']:coordinates['x_finish']] = softmax_tumor_masked

        segmentation_np[coordinates['z_start']:coordinates['z_finish'],
                        coordinates['y_start']:coordinates['y_finish'],
                        coordinates['x_start']:coordinates['x_finish']] = pred_2_np


        segmentation_image = sitk.GetImageFromArray(segmentation_np)
        segmentation_image.CopyInformation(itk_img)


        # Convert nnUNet prediction back to physical space of input scan
        pred_itk_resampled = translate_pred_to_reference_scan_from_file(
            pred                = pm_image,
            reference_scan_path = str(self.ct_image) # check if self.ct_image is the cropped_ROI
        )

        # save prediction to output folder
        sitk.WriteImage(pred_itk_resampled, str(self.heatmap), True)
        sitk.WriteImage(segmentation_image, str(self.segmentation), True)
        subprocess.check_call(["ls", str(self.output_dir_tlm), "-al"])
        subprocess.check_call(["ls", str(self.output_dir_seg), "-al"])

        # if output raw heatmap option is enabled, output the unmasked tumor likelihood map...
        if self.output_raw_heatmap:
            # zero pad the raw heatmap to match the input image dimensions
            pm_raw_image = np.zeros(image_np.shape, dtype=np.float32)
            pm_raw_image[
                coordinates['z_start']:coordinates['z_finish'],
                coordinates['y_start']:coordinates['y_finish'],
                coordinates['x_start']:coordinates['x_finish']
            ] = pred_ensemble  # softmax_tumor (not masked)
            # convert nnUNet prediction back to physical space of input scan
            pred_raw_itk_resampled = translate_pred_to_reference_scan_from_file(
                pred                = pm_raw_image,
                reference_scan_path = str(self.ct_image)
            )
            # save prediction to output folder
            sitk.WriteImage(pred_raw_itk_resampled, str(self.heatmap_raw), True)


    def predict(self, input_dir, output_dir, task="Task103_AllStructures", trainer="nnUNetTrainerV2",
                network="3d_fullres", checkpoint="model_final_checkpoint", folds="0,1,2,3,4", 
                store_probability_maps=True, disable_augmentation=False, 
                disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        cmd_str = " ".join(cmd)
        subprocess.check_call(cmd_str, shell=True)


def translate_pred_to_reference_scan_from_file(pred, reference_scan_path, transpose = False):
    """
    Compatibility layer for `translate_pred_to_reference_scan`
    - pred_path: path to softmax / binary prediction
    - reference_scan_path: path to SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing
    Returns:
    - SimpleITK Image pred_itk_resampled: 
    """
    if transpose:
        pred = pred.T

    # read reference scan and resample reference to spacing of training data
    reference_scan = sitk.ReadImage(reference_scan_path, sitk.sitkFloat32)

    pred_itk = sitk.GetImageFromArray(pred)
    pred_itk.CopyInformation(reference_scan)

    return pred_itk


if __name__ == "__main__":
    PDACDetectionContainer(output_raw_heatmap=False).process()
