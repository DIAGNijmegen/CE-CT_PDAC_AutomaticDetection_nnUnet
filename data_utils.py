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

import numpy as np
import SimpleITK as sitk
import time
import os

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    
    out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image

def GetSequencesLabel(label_array):
    result = []
    last = 0
    for i in range(len(label_array) - 1):
        if (label_array[i + 1] - label_array[i]) > 1:
            sub_array = label_array[last:i + 1]
            last = i + 1
            result.append(sub_array)

    if last != label_array[-1]:
        sub_array = label_array[last:]
        result.append(sub_array)
    return result

def GetROIfromDownsampledSegmentation(full_image, image_low_res, mask_low_res, marginx, marginy, marginz):
    #mask_low_res_np = sitk.GetArrayFromImage(mask_low_res)
    prediction_low_resolution_dilated = sitk.BinaryDilate(mask_low_res, (5,5,1))
    mask_low_res_np = sitk.GetArrayFromImage(prediction_low_resolution_dilated)
    mask_non_zeros = np.nonzero(mask_low_res_np)
    if (np.sum(mask_non_zeros) == 0):
        print('ERROR NO SEGMENTATION')
        return 0
    crop_coordinates = []
    for dim in range(len(mask_non_zeros)):
        unique_array = np.unique(mask_non_zeros[dim])
        seq = GetSequencesLabel(unique_array)
        if len(seq) > 1:
            size_seqs = [len(i) for i in seq]
            for s in range(len(size_seqs)):
                if size_seqs[s] == max(size_seqs):
                    select_seq = seq[s]
                    crop_coordinates.append(
                        [select_seq[0], select_seq[-1]])
                else:
                    print('Segmentation has mistakes')
        else:
            crop_coordinates.append([seq[0][0], seq[0][-1]])

    #Get physical point for crop coordinates
    start_point = [crop_coordinates[2][0], crop_coordinates[1][0], crop_coordinates[0][0]]
    finish_point = [crop_coordinates[2][1], crop_coordinates[1][1], crop_coordinates[0][1]]

    physical_start = image_low_res.TransformIndexToPhysicalPoint((int(start_point[0]), int(start_point[1]), int(start_point[2])))
    physical_finish = image_low_res.TransformIndexToPhysicalPoint((int(finish_point[0]), int(finish_point[1]), int(finish_point[2])))

    full_size = full_image.GetSize()
    coordinates_start_full_image = full_image.TransformPhysicalPointToIndex(physical_start)
    coordinates_end_full_image = full_image.TransformPhysicalPointToIndex(physical_finish)

    # Add margin
    x_start = max(0, coordinates_start_full_image[0] - marginx)
    x_finsh = min(full_size[0], coordinates_end_full_image[0] + marginx)
    y_start = max(0, coordinates_start_full_image[1] - marginy)
    y_finsh = min(full_size[1], coordinates_end_full_image[1] + marginy)
    z_start = max(0, coordinates_start_full_image[2] - marginz)
    z_finsh = min(full_size[2], coordinates_end_full_image[2] + marginz)
    
    coordinates = {
        'x_start': x_start,
        'x_finish': x_finsh,
        'y_start': y_start,
        'y_finish': y_finsh,
        'z_start': z_start,
        'z_finish': z_finsh
    }

    cropped_image = full_image[x_start:x_finsh, y_start:y_finsh, z_start:z_finsh]
    return cropped_image, coordinates

def GetPancreasDilatedMaskFromPrediction(nnUnet_prediction, allStructures):
    prediction_np = sitk.GetArrayFromImage(nnUnet_prediction)
    if allStructures:
        prediction_np[prediction_np==1] = 4
        prediction_np[prediction_np==5] = 4
        prediction_np[prediction_np==7] = 4
        prediction_np[prediction_np!=4] = 0
        prediction_np[prediction_np==4] = 1
    else:
        prediction_np[prediction_np==1] = 2
        prediction_np[prediction_np==2] = 1
    
    prediction_combined = sitk.GetImageFromArray(prediction_np)
    prediction_combined.CopyInformation(nnUnet_prediction)
    prediction_dilated = sitk.BinaryDilate(prediction_combined, (5,5,1))
    return prediction_dilated

def FPreductionPancreasMaskEnsamble(prediction_image_1,prediction_image_2, esamble_pm_numpy, allStructures):

    dilated_prediction1            = GetPancreasDilatedMaskFromPrediction(prediction_image_1, allStructures)
    dilated_prediction2            = GetPancreasDilatedMaskFromPrediction(prediction_image_2, allStructures)
    prediction_pancreas_image_1_np = sitk.GetArrayFromImage(dilated_prediction1)
    prediction_pancreas_image_2_np = sitk.GetArrayFromImage(dilated_prediction2)

    prediction_pancreas_combined   = (prediction_pancreas_image_1_np + prediction_pancreas_image_2_np) / 2.0
    prediction_pancreas_combined   = prediction_pancreas_combined.astype(np.uint8)
    #prediction_pancreas_combined   = prediction_pancreas_combined.T

    softmax_tumor_masked           = esamble_pm_numpy * prediction_pancreas_combined 

    return softmax_tumor_masked

# def FPreductionPancreasMaskEnsamble(prediction_image, pm_numpy, allStructures):

#     dilated_prediction            = GetPancreasDilatedMaskFromPrediction(prediction_image, allStructures)
#     prediction_pancreas_image_np = sitk.GetArrayFromImage(dilated_prediction)

#     prediction_pancreas   = prediction_pancreas_image_np.astype(np.uint8)

#     softmax_tumor_masked           = pm_numpy * prediction_pancreas 

#     return softmax_tumor_masked


def writeSlices(series_tag_values, prefix, new_img, out_dir, i, writer):
    image_slice = new_img[:, :, i]

    # Tags shared by the series.
    list(
        map(
            lambda tag_value: image_slice.SetMetaData(
                tag_value[0], tag_value[1]
            ),
            series_tag_values,
        )
    )

    # Slice specific tags.
    #   Instance Creation Date
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    #   Instance Creation Time
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

    # Setting the type to CT so that the slice location is preserved and
    # the thickness is carried over.
    image_slice.SetMetaData("0008|0060", "CT")

    # (0020, 0032) image position patient determines the 3D spacing between
    # slices.
    #   Image Position (Patient)
    image_slice.SetMetaData(
        "0020|0032",
        "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
    )
    #   Instance Number
    image_slice.SetMetaData("0020,0013", str(i))

    # Write to the output directory and add the extension dcm, to force
    # writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir, prefix + '_' + str(i) + ".dcm"))
    writer.Execute(image_slice)

def ConvertSitkImageToDicomSeries(out_dir, prefix, study_instance_UID, series_instance_UID, series_description, sitk_image, pixel_dtype):
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number,
    # cannot start with zero, and separated by a '.' We create a unique series ID
    # using the date and time. Tags of interest:
    direction = sitk_image.GetDirection()
    series_tag_values = [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        ("0020|000e", series_instance_UID), # Series Instance UID
        ("0020|000d", study_instance_UID), #Study Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),  # Image Orientation
        # (Patient)
        ("0008|103e", series_description),  # Series Description
    ]

    if pixel_dtype == 'float':
        # If we want to write floating point values, we need to use the rescale
        # slope, "0028|1053", to select the number of digits we want to keep. We
        # also need to specify additional pixel storage and representation
        # information.
        rescale_slope = 0.001  # keep three digits after the decimal point
        series_tag_values = series_tag_values + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|1052", "0"),  # rescale intercept
            ("0028|0100", "16"),  # bits allocated
            ("0028|0101", "16"),  # bits stored
            ("0028|0102", "15"),  # high bit
            ("0028|0103", "1"),
        ]  # pixel representation

    # Write slices to output directory
    list(
        map(
            lambda i: writeSlices(series_tag_values, prefix, sitk_image, out_dir, i, writer),
            range(sitk_image.GetDepth()),
        )
    )


