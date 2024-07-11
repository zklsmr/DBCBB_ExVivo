### zklsmr ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from subprocess import run, call
import shlex
import nibabel as nib
import os
import re

def DWI_preprocessing(DWI_vol, pe, b0_peep, outdir, mask=None, clean=True):
    """
    args:
        DWI_vol: path to the DWI volume
        pe: phase encoding direction (RL or LR)
        pe_b0: path to the b0 volume with phase encoding direction pe
        rpe_b0: path to the b0 volume with reverse phase encoding direction
        mask: path to the brain mask
        outdir: path to the output directory
        clean: boolean to remove the temporary files
    returns:
        None
    """

    if mask is not None:
        if not is_binary(mask): # function to check if the mask is binary
            raise ValueError("The mask you provided is not binary.")
    else:
        print("No mask provided. Proceeding without a mask.")

    
    _nametest = outdir.split("/")[-3].split("_")
    dadmah_index = _nametest.index('dadmah')
    _id = [item for item in _nametest[dadmah_index:dadmah_index+4] if item.isdigit()]
    if len(_id) > 0:
        _id = _id[0]


    B0_peep = b0_peep
    #dirs to save# 
    dwi_den = f"{outdir}/{_id}_{_nametest[-2]}_dwi_den.mif"
    dwi_noise = f"{outdir}/{_id}_{_nametest[-2]}_noise.nii.gz"
    dwi_den_unr = f"{outdir}/{_id}_{_nametest[-2]}_dwi_den_unr.mif"
    dwi_den_unr_preproc = f"{outdir}/{_id}_{_nametest[-2]}_dwi_den_unr_preproc.mif"
    dwi_den_unr_preproc_abs = f"{outdir}/{_id}_{_nametest[-2]}_dwi_den_unr_preproc_abs.mif"
    fsl_config = "/data/dadmah/alazak/modelling_exvivo/diff_protocol-alasmar2024/ShellsComps/exvivo_diffusion/cnf.cnf"
    dwi_den_unr_preproc_unb = f"{outdir}/{_id}_{_nametest[-2]}_dwi_den_unr_preproc_unb.mif"
    dwi_bias = f"{outdir}/{_id}_{_nametest[-2]}_bias.nii.gz"

    pe_dir = "RL" if pe == "RL" else "LR"

    if mask is not None:
        dwi_denoise = f"dwidenoise {DWI_vol} {dwi_den} -noise {dwi_noise}"
        dwi_unring_rl = f"mrdegibbs {dwi_den} {dwi_den_unr} -axes 1,2"
        dwi_fsl = f"dwifslpreproc {dwi_den_unr} {dwi_den_unr_preproc}\
            -eddy_mask {mask}\
             -pe_dir {pe_dir} -rpe_pair -se_epi {B0_peep}\
             -eddy_options ' --slm=linear'\
             -topup_options ' --config={fsl_config}'"
        no_zero_vals = f"mrcalc {dwi_den_unr_preproc} 0 -gt 1 -add {dwi_den_unr_preproc_0gt}"
        dwi_unbias = f"dwibiascorrect ants {dwi_den_unr_preproc_0gt} {dwi_den_unr_preproc_unb} -bias {dwi_bias}"

    elif mask is None:
        dwi_denoise = f"dwidenoise {DWI_vol} {dwi_den} -noise {dwi_noise}"
        dwi_unring_rl = f"mrdegibbs {dwi_den} {dwi_den_unr} -axes 1,2"
        dwi_fsl = f"dwifslpreproc {dwi_den_unr} {dwi_den_unr_preproc}\
             -pe_dir {pe_dir} -rpe_pair -se_epi {B0_peep}\
             -eddy_options ' --slm=linear'\
             -topup_options ' --config={fsl_config}'"
        no_zero_vals = f"mrcalc {dwi_den_unr_preproc} -abs {dwi_den_unr_preproc_abs}"
        dwi_unbias = f"dwibiascorrect ants {dwi_den_unr_preproc_abs} {dwi_den_unr_preproc_unb} -bias {dwi_bias}"

    
    
    call(shlex.split(dwi_denoise))
    call(shlex.split(dwi_unring_rl))
    call(shlex.split(dwi_fsl))
    call(shlex.split(no_zero_vals))
    call(shlex.split(dwi_unbias))
    
    
    return


def get_tensor_metrics(DWI_vol, mask, outdir, num_evals = 3):
    """
    args:
        DWI_vol: path to the DWI volume
        mask: path to the brain mask
        outdir: path to the output directory
    returns:
        None
    """
    _nametest = outdir.split("/")[-3].split("_")
    dadmah_index = _nametest.index('dadmah')
    _id = [item for item in _nametest[dadmah_index:dadmah_index+4] if item.isdigit()]
    if len(_id) > 0:
        _id = _id[0]

    DTI = f"{outdir}/{_id}_{_nametest[-2]}_DTI.nii.gz"
    FA = f"{outdir}/{_id}_{_nametest[-2]}_FA.nii.gz"
    MD = f"{outdir}/{_id}_{_nametest[-2]}_MD.nii.gz"
    RD = f"{outdir}/{_id}_{_nametest[-2]}_RD.nii.gz"
    AD = f"{outdir}/{_id}_{_nametest[-2]}_AD.nii.gz"

    eigen_vals = f"{outdir}/{_id}_{_nametest[-2]}_eigenvals.nii.gz"
    eigen_vecs = f"{outdir}/{_id}_{_nametest[-2]}_eigenvecs.nii.gz"
    
    dwi_tensor = f"dwi2tensor {DWI_vol} {DTI} -iter 10 -mask {mask}"
    tensor_metrics = f"tensor2metric {DTI} -mask {mask}\
         -fa {FA} -adc {MD} -rd {RD} -ad {AD}\
             -vector {eigen_vecs} -value {eigen_vals} -num {num_evals} -modulate eigval"

    call(shlex.split(dwi_tensor))
    call(shlex.split(tensor_metrics))



def is_binary(mask):
    mask_data = nib.load(mask).get_fdata()
    mask_data = np.unique(mask_data)
    if len(mask_data) > 1:
        return True
    else:
        return False
    


def resample_to_b0(b0, mask):
    """
    args:
        b0: path to the b0 volume
        mask: path to the brain mask
    returns:
        None
    """
    out_dir = mask.split("BISON")[0]
    resampled_mask = f"{out_dir}/resampled_mask.nii.gz"
    resample_cmd = f"itk_resample {mask} --like {b0} --labels {resampled_mask}"
    call(shlex.split(resample_cmd))


def extract_labels(mask, extract, outdir, labels_to_combine=None):
    """
    args:
        mask: path to the mask
        extract: list of labels to extract
        outdir: path to the output directory
        labels_to_combine: list of labels to combine into one file
    returns:
        None
    """
    mask_data = nib.load(mask).get_fdata()
    labels = np.unique(mask_data)
    labels = labels[labels != 0] 
    labels = [label for label in labels if label in extract]
    
    for label in labels:
        label_mask = np.zeros_like(mask_data)
        label_mask[mask_data == label] = 1
        nib.save(nib.Nifti1Image(label_mask, affine=nib.load(mask).affine), f"{outdir}/label_{label}.nii.gz")
    
    if labels_to_combine:
        combined_mask = np.zeros_like(mask_data)
        for label in labels_to_combine:
            combined_mask[mask_data == label] = 1  # Set combined mask to 1 for each label
        nib.save(nib.Nifti1Image(combined_mask, affine=nib.load(mask).affine), f"{outdir}/combined_labels.nii.gz")

def extract_labels_YZ(mask, extract, outdir, labels_to_combine=None):
    """
    args:
        mask: path to the mask
        extract: list of labels to extract
        outdir: path to the output directory
        labels_to_combine: list of labels to combine into one file
    returns:
        None
    """
    mask_data = nib.load(mask).get_fdata()
    labels = np.unique(mask_data)
    labels = labels[labels != 0] 
    labels = [label for label in labels if label in extract]
    
    for label in labels:
        label_mask = np.zeros_like(mask_data)
        label_mask[mask_data == label] = 1
        nib.save(nib.Nifti1Image(label_mask, affine=nib.load(mask).affine), f"{outdir}/label_{label}.nii.gz")
    
    if labels_to_combine:
        combined_mask = np.zeros_like(mask_data)
        for label in labels_to_combine:
            combined_mask[mask_data == label] = label  # Set combined mask to 1 for each label
        nib.save(nib.Nifti1Image(combined_mask, affine=nib.load(mask).affine), f"{outdir}/combined_labels_YZ.nii.gz")



def threshold_merge_b0(b0, mask, outdir, b0_rpe=None, pe=None):
    """
    args:
        b0: path to the b0 volume
        mask: path to the brain mask
        outdir: path to the output directory
        b0_rpe: path to the b0 volume with reverse phase encoding direction
        pe: phase encoding direction
    returns:
        thresholded_b0: path to the thresholded b0 volume
        thresholded_b0_rpe: path to the thresholded b0 volume with reverse phase encoding direction
    """
    if pe is not None and 'pe' in b0:
        thresholded_b0 = f"{outdir}/thresholded_b0.nii.gz"
        threshold_cmd = f"fslmaths {b0} -mul {mask} {thresholded_b0}"
        call(shlex.split(threshold_cmd))
        if b0_rpe is not None:
            thresholded_b0_rpe = f"{outdir}/thresholded_b0_rpe.nii.gz"
            threshold_cmd_rpe = f"fslmaths {b0_rpe} -mul {mask} {thresholded_b0_rpe}"
            # call(shlex.split(threshold_cmd_rpe))

            # Merge the thresholded b0 volumes
            merged_b0 = f"{outdir}/merged_b0.nii.gz"
            merge_cmd = f"fslmerge -t {merged_b0} {thresholded_b0} {thresholded_b0_rpe}"
            # call(shlex.split(merge_cmd))
            
            merge_cmd_no_threshold = f"fslmerge -t {outdir}/merged_b0_no_threshold.nii.gz {b0} {b0_rpe}"
            call(shlex.split(merge_cmd_no_threshold))



def mask_metric(metric, mask):
    """
    args:
        metric: path to the metric volume
        mask: path to the brain mask
        outdir: path to the output directory
    returns:
        masked_metric: path to the masked metric volume
    """
    masked_metric = os.path.join(os.path.dirname(metric), f"{os.path.basename(metric).split('.')[0]}_masked.nii.gz")
    mask_cmd = f"fslmaths {metric} -mul {mask} {masked_metric}"
    call(shlex.split(mask_cmd))
    return masked_metric



def fit_response(dwi, mask, outdir):
    """
    args:
        dwi: path to the DWI volume
        mask: path to the brain mask
        outdir: path to the output directory
    returns:
        None
    """
    response_wm = f"{outdir}/response_wm.txt"
    response_gm = f"{outdir}/response_gm.txt"
    response_csf = f"{outdir}/response_csf.txt"
    response_cmd = f"dwi2response dhollander {dwi} {response_wm} {response_gm} {response_csf}"
    call(shlex.split(response_cmd))


def fit_fODF(dwi, outdir, mask = None):
    """
    args:
        dwi: path to the DWI volume
        mask: path to the brain mask
        outdir: path to the output directory
    returns:
        None
    """
    if mask is not None:
        if not is_binary(mask): # function to check if the mask is binary
            raise ValueError("The mask you provided is not binary.")
    else:
        print("No mask provided. Proceeding without a mask.")

    _nametest = outdir.split("/")[-3].split("_")
    dadmah_index = _nametest.index('dadmah')
    _id = [item for item in _nametest[dadmah_index:dadmah_index+4] if item.isdigit()]
    if len(_id) > 0:
        _id = _id[0]

    fODF = f"{outdir}/{_id}_{_nametest[-2]}_fODF.nii.gz"


    mrinfo_64_command = f'mrinfo {dwi} -size'
    mrinfo_64_output = subprocess.check_output(shlex.split(mrinfo_64_command)).decode('utf-8').strip()
    if int(mrinfo_64_output.split(" ")[-1]) > 65: #choose 40 at random, kinda screens for 30
        dwi_fodf = f"dwi2fod msmt_csd {dwi}\
             {outdir}/response_wm.txt  {outdir}/wmfod.mif\
                 {outdir}/response_gm.txt  {outdir}/gm.mif\
                     {outdir}/response_csf.txt  {outdir}/csf.mif"
    else:

        dwi_fodf = f"dwi2fod csd {dwi}\
             {outdir}/response_wm.txt  {outdir}/wmfod.mif"\


    call(shlex.split(dwi_fodf))
    call(shlex.split(fodf_peaks))
    call(shlex.split(fodf_metrics))


def make_tractogram(img, seed, mask, n_tracks=5000,seeds=int(1e7), min_length=50,max_length=2000, force=False):
    """
    args:
        img: path to the DWI volume
        mask: path to the brain mask
        n_tracks: number of tracks to generate
    returns:
        None
    """
    tractogram_fname = f"{os.path.dirname(img)}/{os.path.basename(img).split('.')[0]}_tractogram.tck"

    if force:
        tckgen_cmd = f"tckgen -algorithm Tensor_Prob {img}\
            -select {n_tracks} -seeds {seeds} -minlength {min_length} -maxlength {max_length}\
            -seed_image {seed} -mask {mask} {tractogram_fname} -force"
    else:
        tckgen_cmd = f"tckgen -algorithm Tensor_Prob {img}\
            -select {n_tracks} -seeds {seeds} -minlength {min_length} -maxlength {max_length}\
            -seed_image {seed} -mask {mask} {tractogram_fname}"
    
    call(shlex.split(tckgen_cmd))



def organize_files(indir):
    '''
    moves all masks to a new subdirectory called indir/masks
    moves all B0 volumes to a new subdirectory called indir/B0   
    '''

    masks_dir = f"{indir}/masks"
    B0_dir = f"{indir}/B0"
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(B0_dir, exist_ok=True)

    for file in os.listdir(indir):
        if os.path.isdir(os.path.join(indir, file)):
            continue
        else:
            if "label" in file or "mask" in file or "combined" in file:
                os.rename(f"{indir}/{file}", f"{masks_dir}/{file}")
            elif "b0" in file:
                os.rename(f"{indir}/{file}", f"{B0_dir}/{file}")




def prepare_transormations(xfm_dir, outdir, to_icbm=False):
    '''
    Extracts the linear and non-linear transformations from the xfm directory
    and concatenates them to a single transformation file
    Copy the warp too
    '''
    from glob import glob
    

    xfm_lin = glob(f"{xfm_dir}/stx_lin/*t1_to_icbm.xfm")[0]
    xfm_nl = glob(f"{xfm_dir}/stx_nlin/*inv_nlin_0_inverse_NL.xfm")[0]
    warp_nl = glob(f"{xfm_dir}/stx_nlin/*inv_nlin_0_inverse_NL*mnc")[0]

    xfm_out = f"{outdir}/transformations"
    os.makedirs(xfm_out, exist_ok=True)

    cp_xfm_lin = f"cp {xfm_lin} {xfm_out}/{xfm_lin.split('/')[-1]}"
    cp_xfm_nl1 = f"cp {xfm_nl} {xfm_out}/{xfm_nl.split('/')[-1]}"
    cp_xfm_nl2 = f"cp {warp_nl} {xfm_out}/{warp_nl.split('/')[-1]}"

    inverse_xfm_lin = f"xfminvert {xfm_out}/{xfm_lin.split('/')[-1]} {xfm_out}/{xfm_lin.split('/')[-1].split('.')[0]}_inverse.xfm"
    
    #to icbm or just template
    if to_icbm:
        xfm_to_icbm = "/data/dadmah/ex_vivo_DBCBB_Preprocessing/DBCBB_Template/Proc/low_res_template_t1_to_icbm_0_inverse_NL.xfm"
        warp_to_icbm = "/data/dadmah/ex_vivo_DBCBB_Preprocessing/DBCBB_Template/Proc/low_res_template_t1_to_icbm_0_inverse_NL_grid_0.mnc"
        cp_warp = f"cp {warp_to_icbm} {xfm_out}/{warp_to_icbm.split('/')[-1]}"
        concat_xfm = f"xfmconcat\
            {xfm_to_icbm} {xfm_out}/{xfm_nl.split('/')[-1]} {xfm_out}/{xfm_lin.split('/')[-1].split('.')[0]}_inverse.xfm\
                {xfm_out}/InverseNL_invLin.xfm"
        print(concat_xfm)
    
    else:
        concat_xfm = f"xfmconcat {xfm_out}/{xfm_nl.split('/')[-1]} {xfm_out}/{xfm_lin.split('/')[-1].split('.')[0]}_inverse.xfm\
            {xfm_out}/InverseNL_invLin.xfm"


    call(shlex.split(cp_xfm_lin))
    call(shlex.split(cp_xfm_nl1))
    call(shlex.split(cp_xfm_nl2))
    call(shlex.split(inverse_xfm_lin))
    call(shlex.split(cp_warp))
    call(shlex.split(concat_xfm))



def extract_transformation_matrix(xfm_file,txt_out):
    ''' 
    Extracts the transformation matrix from a .xfm file and saves it as a .txt file
    '''


    with open(xfm_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Linear_Transform" in line:
               matrix_lines = lines[i+1:i+4]
               matrix = [list(map(str, line.split())) for line in matrix_lines]
               matrix = [list(map(str, line.split())) for line in matrix_lines]
               matrix = np.array(matrix).reshape(3, 4)
               matrix[2,3] = matrix[2,3].replace(";", "")
               np.savetxt(txt_out, matrix, fmt='%s')

               return matrix


def niftify_warp(warp_mnc, outdir):
    warp_nii = f"{outdir}/{os.path.basename(warp_mnc).split('.')[0]}.nii.gz"
    mnc2nii_cmd = f"mnc2nii {warp_mnc} {warp_nii}"
    call(shlex.split(mnc2nii_cmd))
    return warp_nii


def resample_2template(img, xfm_dir, template, outdir):

    resampled_img = f"{outdir}/{os.path.basename(img).split('.')[0]}_resampled.nii.gz"
    mrtransofmr_cmd = f"mrtransform {img} -linear {xfm_dir}/InverseNL_invLin.xfm {resampled_img} -template {template} -warp {xfm_dir}/warp.nii.gz"

    call(shlex.split(mrtransofmr_cmd))
    return resampled_img