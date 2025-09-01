
%% <PU2_Attenuation_masking_display_handheld.m>
%
%
% Display the MEprobe attenuation results with the masking.
% Based on Ken's codes: segment_widefield_atten_mat_MEprobeUI.m
%
% Peijun Gong, 2023.08.21
%==========================================================================

%% (1) Load the saved attenuation and OCT data.

% Only keep the aligned scans.
clearvars -except atten oct;

% BT006M Location 3.
%--------------------------------------------------------------------------
% % Input the name of the scan.
% scan_name = 'BT006M-L3-NB-0003';
% % Load the data for display.
% load('2022_06_13_BT006M_Location_3_L3-000-NB_Attenuation_z58.mat');
% load('2022_06_13_BT006M_Location_3_L3-000-NB_OCT_z58.mat');
%--------------------------------------------------------------------------

% BT012M Location 2.
%--------------------------------------------------------------------------
% % Input the name of the scan.
% scan_name = 'BT012M-L2-LU-0001';
% % Load the data for display.
% load('2023_05_15_BT012M_Location_2_002-LU_Attenuation_z58.mat');
% load('2023_05_15_BT012M_Location_2_002-LU_OCT_z58.mat');
%--------------------------------------------------------------------------

% B159M Location 1.
%--------------------------------------------------------------------------
% Input the name of the scan.
scan_name = 'B159M-L1-YK-0001';
% Load the data for display.
load('2020_07_31_B159M_Location_1_L1-000-YK_Attenuation_z58.mat');
load('2020_07_31_B159M_Location_1_L1-000-YK_OCT_z58.mat');
%--------------------------------------------------------------------------


%% (2) Display the scans.

%--------------------------------------------------------------------------
% Reset the filename for saving.
scan_name = 'B159M-L1-YK-0007';

% Depth for the en face images.
z_idx = 7;
%--------------------------------------------------------------------------

% Get the en face images.
log_OCT = squeeze(oct(:, :, z_idx));
mu_t = squeeze(atten(:, :, z_idx));

% Set the resholds for the OCT and attenuation images.
oct_limits = [0 29];
mu_t_limits = [0 10];

% Remove NAN in OCT.
log_OCT(isnan(log_OCT)) = oct_limits(1);

% Display the images.
figure; imshow(log_OCT, oct_limits, 'XData', [0 6], 'YData', [0 6]);
colormap(gray); colorbar;
figure; imshow(mu_t, mu_t_limits, 'XData', [0 6], 'YData', [0 6]);
colormap(parula); colorbar;


%% Save the OCT and mut image with export_fig.

% Add the path for the export_fig scripts.
addpath('D:\peijun\Codes\Export_Figures');

% Save OCT.
export_fig([scan_name, '_MEprobe_OCT_z_58_export_fig.pdf'], '-native', '-transparent', '-a1', '-q101');
export_fig([scan_name, '_MEprobe_OCT_z_58_export_fig.png'], '-native', '-transparent', '-a1', '-q101');

% Save mut.
export_fig([scan_name, '_MEprobe_mut_z_58_export_fig.pdf'], '-native', '-transparent', '-a1', '-q101');
export_fig([scan_name, '_MEprobe_mut_z_58_export_fig.png'], '-native', '-transparent', '-a1', '-q101');


%% (2) Display the attenuation image after masking NANs.

% Segment the attenuation image based on areas of NAN attenuation.
nan_mu_t_mask_xy = true(size(mu_t));
nan_mu_t_mask_xy(isnan(mu_t)) = false;
im_nan_mu_t = nan_mu_t_mask_xy;

% Convert the mask to transparency.
im_alpha = ind2rgb(im_nan_mu_t, gray(2));

% Convert to clipped grayscale and then to 8-bit indexed image.
im_mu_t_ind = gray2ind(mat2gray(mu_t, [mu_t_limits(1) mu_t_limits(2)]), 256);
              
% Convert to RGB so they can be combined.    
im_mu_t = ind2rgb(im_mu_t_ind, parula(256));
im_background = zeros(size(im_mu_t) );

% Combine images.
im_attenuation = im_background.*(1-im_alpha) + im_mu_t .* im_alpha;

% Convert OCT to clipped grayscale, then to 8-bit indexed image
im_oct_ind = gray2ind(mat2gray(log_OCT, oct_limits), 256);

% Convert OCT to RGB (truecolor) so they can be combined.
im_oct = ind2rgb(im_oct_ind, gray(256));

% Display and save the image.
im_mu_t_filt = mu_t;
im_mu_t_filt(isnan(im_mu_t_filt)) = 0;
im_mu_t_pre = ind2rgb(gray2ind(mat2gray(im_mu_t_filt, [mu_t_limits(1) mu_t_limits(2)]), 256), parula(256));
figure; imshow(im_mu_t_pre, [], 'XData', [0 45], 'YData', [0 45]); colorbar;
% imwrite(im_mu_t_pre, [scan_name, '_MEprobe_mut_z_58_imwrite_nan_masked_blue.png']);

figure; imshow(im_attenuation, [], 'XData', [0 6], 'YData', [0 6]); colorbar;
% imwrite(im_attenuation, [scan_name, '_MEprobe_mut_z_58_imwrite_nan_masked_black.png']);

figure; imshow(im_oct, [], 'XData', [0 6], 'YData', [0 6]); colorbar;
% imwrite(im_oct, [scan_name, '_MEprobe_OCT_z_58_imwrite.png']);


%% (3) Mask the attenuation image.

% Add the path for the masking codes.
addpath('D:\peijun\Codes');

% Set the OCT limits for generating the mask.
oct_mask_limits = [0 8];

% Get to the mask.
seg_mask_xy = StromaClassifier_v5_PG(log_OCT, oct_mask_limits);

% figure; imshow(log_OCT', oct_limits, 'XData', [0 6], 'YData', [0 6]); colormap(gray);
figure; imshow(seg_mask_xy,  [0 1], 'XData', [0 6], 'YData', [0 6]); colormap(gray)

% Combine the segmentation mask with the data mask.
im_seg_mu_t = seg_mask_xy & im_nan_mu_t;

% Dislay the mask.
figure; imshow(im_seg_mu_t , [0 1], 'XData', [0 45], 'YData', [0 45]);
    
% Convert OCT to clipped grayscale, then to 8-bit indexed image
im_oct_ind = gray2ind(mat2gray(log_OCT, oct_limits), 256);

% Convert OCT to RGB (truecolor) so they can be combined.
im_oct = ind2rgb(im_oct_ind, gray(256));
    
% Convert attenuation to clipped grayscale, then to 8-bit indexed image
im_mu_t_ind = gray2ind(mat2gray(mu_t, [mu_t_limits(1) mu_t_limits(2)]), 256);

% Convert to attenuation to RGB (truecolor).
im_mu_t = ind2rgb(im_mu_t_ind, parula(256));

% Convert the mask to transparency (only 0 or 1).
im_alpha = ind2rgb(im_seg_mu_t, gray(2));

% Combine images
im_overlay = im_oct.* (1-im_alpha) + im_mu_t .* im_alpha;
figure; imshow(im_overlay, [], 'XData', [0 45], 'YData', [0 45]);

% Fully maksed images
im_masked = im_mu_t.* im_alpha;
figure; imshow(im_masked, [], 'XData', [0 45], 'YData', [0 45]);

%%
% Save the image and mask.
imwrite(im_overlay, [scan_name, '_MEprobe_mut_z_58_imwrite_OCT_combined.png']);
imwrite(im_masked, [scan_name, '_MEprobe_mut_z_58_imwrite_fully_masked.png']);
save([scan_name, '_MEprobe_results'], 'log_OCT','mu_t', 'oct_mask_limits', 'seg_mask_xy', 'im_seg_mu_t');

%% Pass the save the mask infomration for subsequent analysis.
% mut_mask = zeros(size(atten));

mut_mask(:,:,z_idx) = im_seg_mu_t;
z_idx
% Save the mask.
% save([scan_name, 'mut_mask'],'mut_mask');

%%
 
%% End of the function.


