import os
import SimpleITK
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


class PDACDetectionContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir  = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir  = Path("/opt/algorithm/nnunet/results")

        # input / output paths
        self.ct_ip_dir         = Path("/input/images/pancreas-ct")
        self.output_dir        = Path("/output/images/pancreas-ct-heatmap")
        self.ct_image          = Path(self.ct_ip_dir).glob("*.mha")
        self.heatmap           = self.output_dir / "heatmap.mha"

        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(os.listdir(self.ct_ip_dir))


        for fn in os.listdir(self.ct_ip_dir):
            if ".mha" in fn: self.ct_image = os.path.join(self.ct_ip_dir, fn)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load CT scan and Generate Heatmap for Pancreas Cancer  
        """
        # move input images to nnUNet format with __0000.nii.gz 
        newpath_ct = str(self.nnunet_input_dir / "scan_0000.nii.gz")
        itk_img    = sitk.ReadImage(self.ct_image, sitk.sitkFloat32)
        # sitk.WriteImage(itk_img, newpath_ct)
        sitk.WriteImage(itk_img, newpath_ct)
        #atomic_image_write(self.ct_image, str(newpath_ct), useCompression=True)
        

        # Predict using nnUNet ensemble, averaging multiple restarts
        pred_ensemble = None
        ensemble_count = 0
        for trainer in [
            "nnUNetTrainerV2_Loss_CE_checkpoints",
            "nnUNetTrainerV2_Loss_CE_checkpoints2",
        ]:
            self.predict(
                task="Task103_AllStructures",
                trainer=trainer
            )
            pred_path = str(self.nnunet_output_dir / "scan.npz")
            pred = np.load(pred_path)['softmax'][1].astype('float32')
            os.remove(pred_path)
            if pred_ensemble is None:
                pred_ensemble = pred
            else:
                pred_ensemble += pred
            ensemble_count += 1

        # Average the accumulated confidence scores
        pred_ensemble /= ensemble_count

        # Convert nnUNet prediction back to physical space of input scan 
        pred_itk_resampled = translate_pred_to_reference_scan_from_file(
            pred = pred_ensemble,
            reference_scan_path = str(self.ct_image)
        )
  
        # save prediction to output folder
        #atomic_image_write(pred_itk_resampled, str(self.heatmap), useCompression=True)
        sitk.WriteImage(pred_itk_resampled, str(self.heatmap))
        subprocess.check_call(["ls", str(self.output_dir), "-al"])

    def predict(self, task="Task103_AllStructures", trainer="nnUNetTrainerV2",
                network="3d_fullres", checkpoint="model_final", folds="0,1,2,3,4", 
                store_probability_maps=True, disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_input_dir),
            '-o', str(self.nnunet_output_dir),
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

        subprocess.check_call(cmd)


def translate_pred_to_reference_scan_from_file(pred_path, reference_scan_path):
    """
    Compatibility layer for `translate_pred_to_reference_scan`
    - pred_path: path to softmax / binary prediction
    - reference_scan_path: path to SimpleITK image to which the prediction should be resampled and resized
    - out_spacing: spacing to which the reference scan is resampled during preprocessing
    Returns:
    - SimpleITK Image pred_itk_resampled: 
    """
    # read softmax prediction
    pred = np.load(pred_path)['softmax'][1].astype('float32')

    # read reference scan and resample reference to spacing of training data
    reference_scan = sitk.ReadImage(reference_scan_path, sitk.sitkFloat32)

    pred_itk = sitk.GetImageFromArray(pred)
    pred_itk.CopyInformation(reference_scan)

    return pred_itk


if __name__ == "__main__":
    PDACDetectionContainer().process()



