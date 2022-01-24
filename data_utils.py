import numpy as np
import SimpleITK as sitk

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
    mask_low_res_np = sitk.GetArrayFromImage(mask_low_res)
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
    prediction_pancreas_combined   = prediction_pancreas_combined.T

    softmax_tumor_masked           = esamble_pm_numpy * prediction_pancreas_combined 

    return softmax_tumor_masked
