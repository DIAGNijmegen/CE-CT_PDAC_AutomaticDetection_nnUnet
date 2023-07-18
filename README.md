# Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography

#### Please cite the following publication when using this algorithm:

> #### Alves N, Schuurmans M, Litjens G, Bosma JS, Hermans J, Huisman H. Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography. Cancers (Basel). 2022 Jan 13;14(2):376. doi: 10.3390/cancers14020376. PMID: 35053538; PMCID: PMC8774174.

### Summary
This algorithm produces a tumor likelihood heatmap for the presence of pancreatic ductal adenocarcinoma (PDAC) in an input venous-phase contrast-enhanced computed tomography scan (CECT). Additionally, the algorithm segments multiple surrounding anatomical structures such as the pancreatic duct, common bile duct, veins, and arteries. The heatmap and segmentations are resampled to the same spatial resolution and physical dimensions as the input CECT image for easier visualization.

### Mechanism
This is a fully-automatic deep-learning-based framework for PDAC detection. The process starts by obtaining a coarse pancreas segmentation using a low-resolution nnU-net, which is used to extract a volume of interest around the pancreas region from the input scan. Then, this smaller volume of interest is fed to 10 independent nnUnet models (5-fold cross-validation with two restarts), which are ensembled to produce the final PDAC tumor likelihood map. The nnUnet also outputs the segmentation maps for seven different anatomical structures: 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, and 7-renal vein.

For more information, please refer to the publication [Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8774174/).

### Uses and Directions
* This algorithm was developed for research purposes only. This algorithm is intended to be used only on venous-phase CECT examinations of patients with clinical suspicion of PDAC. This algorithm should not be used in different patient demographics.

* Benefits: Automatic detection and localization of PDAC with additional information regarding surrounding anatomy.

* Target Population: This algorithm was trained on a cohort of 242 patients from Radboud University Medical centre, of which 119 had pathologically confirmed PDAC of the pancreatic head.

* General use: This model is intended to be used by radiologists for predicting PDAC in venous-phase CECT scans. The model is not diagnostic for cancer and is not meant to guide or drive clinical care. This model is intended to complement other pieces of patient information in order to determine the appropriate follow-up recommendation.

* Appropriate decision support: The model identifies lesion X as at a high risk of being malignant. The referring radiologist reviews the prediction along with other clinical information and decides the appropriate follow-up recommendation for the patient.

* Before using this model: Test the model retrospectively and prospectively on a diagnostic cohort that reflects the target population that the model will be used upon to confirm the validity of the model within a local setting.

* Safety and efficacy evaluation: To be determined in a clinical validation study.

### Warnings
* Risks: Even if used appropriately, clinicians using this model can misdiagnose cancer. Delays in cancer diagnosis can lead to metastasis and mortality. Patients who are incorrectly treated for cancer can be exposed to risks associated with unnecessary interventions and treatment costs related to follow-ups.

* Generalizability: This model was not trained on scans of PDAC patients with tumors in the body and tail of the pancreas. Hence it is susceptible to faulty predictions and unintended behaviour when presented with such cases.

* Clinical rationale: The model is not interpretable and does not provide a rationale for therisk scores, beyond the localization structures presented in its output heatmap. Clinicians are expected to place the model output in context with other clinical information to make the final determination of diagnosis.

* Discontinue use if: Clinical staff raise concerns about the utility of the model for the intended use-case or large, systematic changes occur at the data level that necessitates re-training of the model.

