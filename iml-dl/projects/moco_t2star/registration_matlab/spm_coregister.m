function spm_coregister(fixed, moving, varargin)
    % spm_coregister.m
    %
    % INPUTS:
    % fixed_image   - Path to the reference image (fixed image)
    % moving_image  - Path to the source image (moving image)
    % other_images  - Cell array of paths to additional images to transform

    % Initialize SPM
    spm('defaults', 'FMRI');
    spm_jobman('initcfg');

    other_images = varargin;

    % Batch for coregistration
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {fixed}; % Reference image
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {moving}; % Source image
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = other_images; % Other images to transform
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'ncc'; % Normalized cross-correlation (intra-modality)
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2]; % Sampling for optimization (coarse to fine)
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = ...
        [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001]; % Tolerance
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7]; % Smoothing for the cost function
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4; % 4th-degree B-spline interpolation
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0]; % No wrapping
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0; % No masking
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r'; % Prefix for registered files

    % Run the batch
    spm_jobman('run', matlabbatch);
end

