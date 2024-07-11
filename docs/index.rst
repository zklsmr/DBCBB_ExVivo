Welcome to DBCBB_ExVivo's documentation!
========================================

.. module:: DBCBB_ExVivo
    :synopsis: A set of functions to perform large-scale ex-vivo imaging preprocessing. 


This documentation details the preprocessing tools developed:
-------------------------------------------------------------
   - Specifically for data preparation, preprocessing, and analysis of the Douglas Bell Canada Brain Bank ex-vivo imaging project.
   - Generally for the preprocessing of all large-scal ex-vivo imaging datasets (existing and to be acquired). 

-------------------------------------------------------------

Currently in 0.1.0alpha, the tools are in development and are not yet fully functional.
-------------------------------------------------------------
This version can only be used with ex-vivo diffusion imaging data, and can perform diffusion tensor fitting, fibre orientation distribution fitting, and determinstic, 
probabilsitic, and constrained spherical tractography.


Reference Us!
========================================
If you use the data or the toolbox package in your research, please cite the following paper:


      Dadar, M., Sanches, L., Fouquet, J., Moqadam, R., Alasmar, Z., Mirault, D., Maranzano, J., Mechawar, N., Chakravarty, M., & Zeighami, Y. 
      (2024). The Douglas Bell Canada Brain Bank Post-mortem Brain Imaging Protocol (p. 2024.02.27.582303). bioRxiv. https://doi.org/10.1101/2024.02.27.582303




Table of Contents:
===================

.. toctree::
   :maxdepth: 1
   :caption: Modalities

   modalities/DiffusionMRI.rst
   modalities/StructuralMRI.rst
   modalities/QuantitativeMRI.rst