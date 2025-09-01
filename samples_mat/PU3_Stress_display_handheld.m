
%% <PU3_Stress_display_handheld.m>
%
%
% Display the MEprobe stress maps.
%
% Peijun Gong, 2023.08.22
%==========================================================================

%% (1) Load the saved stress data.

% BT006M Location 3.
%--------------------------------------------------------------------------
% % Input the name of the scan.
% scan_name = 'BT006M-L3-NB-0003';
% 
% % Load the data for display.
% load('2022_06_13_BT006M_Location_3_L3-000-NB_Stress.mat');
%--------------------------------------------------------------------------

% BT012M Location 2.
%--------------------------------------------------------------------------
% % Input the name of the scan.
% scan_name = 'BT012M-L2-LU-0001';
% 
% % Load the data for display.
% load('2023_05_15_BT012M_Location_2_002-LU_Stress.mat');
%--------------------------------------------------------------------------


% BT001M (B159M) Location 1.
%--------------------------------------------------------------------------
% Input the name of the scan.
scan_name = 'B159M-L1-YK-0001';

% Load the data for display.
load('2020_07_31_B159M_Location_1_L1-000-YK_Stress_New.mat');
%--------------------------------------------------------------------------

%% (2) Display the scans.

%--------------------------------------------------------------------------
% Reset the filename for saving.
scan_name = 'B159M-L1-YK-0007';

% Depth for the en face images.
z_idx = 7;
%--------------------------------------------------------------------------

% Set the thresholds for the stress images.
stress_limits = [0 log10(100)];

% Get the en face data.
ef_stress = squeeze(stress(:, :, z_idx));

% Get the logarithmic data.
ef_stress_log = log10(-ef_stress);

% Display the images.
figure;
imshow(ef_stress_log', [stress_limits(1) stress_limits(2)], 'XData', [0 6], 'YData', [0 6]);
colormap(jet); colorbar;


%% (3) Set up the smoothing window.

% Add path of scripts.
addpath('D:\peijun\Codes');

% Resolution of the OCT scanner.
lateral_res = 14e-3;  % In mm.

% Convert FWHM into equivalent gaussian standard deviation.
sigma_x = oct_filter_sigma(lateral_res, 364, 6);
sigma_y = oct_filter_sigma(lateral_res, 365, 6);

% Generate the filtering kernels.
smooth_kernel = gaussian_filter(2*sigma_x, 4*sigma_y);

% Smooth the logaritmic stress.
ef_stress_sm = (conv2(ef_stress_log, smooth_kernel, 'same'));

% Display the smoothed images.
figure;
imshow(ef_stress_sm', [stress_limits(1) stress_limits(2)], 'XData', [0 6], 'YData', [0 6]);
colormap(jet); colorbar;


%% Save the stress image with export_fig.

% Add the path for the export_fig scripts.
addpath('D:\peijun\Codes\Export_Figures');

% Save stress.
export_fig([scan_name, '_MEprobe_stress_sm_export_fig.pdf'], '-native', '-transparent', '-a1', '-q101');
export_fig([scan_name, '_MEprobe_stress_sm_export_fig.png'], '-native', '-transparent', '-a1', '-q101');

 
%% End of the function.


