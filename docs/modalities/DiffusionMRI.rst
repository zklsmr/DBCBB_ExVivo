Ex-vivo Diffusion MRI Preprocessing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DWI_preprocessing**:

    args:
        - DWI_vol: path to the DWI volume    
        - pe: phase encoding direction        
        - pe_b0: path to the b0 volume with phase encoding direction pe     
        - rpe_b0: path to the b0 volume with reverse phase encoding direction     
        - mask: path to the brain mask        
        - outdir: path to the output directory        
        - clean: boolean to remove the temporary files        
    returns:
        - None

    Given a DWI volume, this function performs the following preprocessing steps:

    - Denoising the DWI volume using the MRtrix3 `dwidenoise` command.  
    - Estimating the noise level in the DWI volume using the MRtrix3 `dwinoise` command.    
    - Unringing the DWI volume to remove Gibbs ringing artifacts using the MRtrix3 `mrdegibbs` command. 
    - Preprocessing the unringed DWI volume by correcting for intensity non-uniformity using the MRtrix3 `dwibiascorrect` command.  
    - Computing the absolute value of the preprocessed DWI volume using the MRtrix3 `mrcalc` command.   
    - Applying the unbiased intensity normalization to the preprocessed DWI volume using the MRtrix3 `mtnormalise` command. 
    - Estimating the bias field in the preprocessed DWI volume using the ANTs `N4` command.    


- **get_tensor_metrics**:

    args:
        - DWI_vol: path to the DWI volume 
        - mask: path to the brain mask    
        - outdir: path to the output directory    
        - num_evals: number of eigenvalues to output  
    returns:
        - None

    After preprocessing, this function computes the diffusion eigenvalues and eigenvectors, and tensor metrics 
    (FA, MD, AD, RD) using the MRtrix3 `dwi2tensor` and `tensor2metric` commands.


- **fit_response**:

    args:
        - dwi: path to the DWI volume 
        - mask: path to the brain mask    
        - outdir: path to the output directory 
    returns:
        - None

    This function estimates the response function of the DWI volume using the MRtrix3 `dwi2response` command.
    It will output the response functions for white matter, gray matter, and cerebrospinal fluid.
    Currently, the function uses the `dhollander` algorithm to estimate the response functions.