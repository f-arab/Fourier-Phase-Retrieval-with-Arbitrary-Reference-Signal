clc
close all
clear variables

%% Signal
addpath(genpath('data/EMNIST'))

dataset_letters = load('emnist-letters');
dataset_letters_images = dataset_letters.dataset.train.images;
total_chars = size(dataset_letters_images,1);
% 59
rng(59)
x1_idx = randi(total_chars,1);
x2_idx = randi(total_chars,1);

x1 = im2double(reshape(dataset_letters_images(x1_idx,:),28,28));
x2 = im2double(reshape(dataset_letters_images(x2_idx,:),28,28));

xt = [x1 x2];
%% First Scenario: All the methods work

known_reference = zeros(size(xt));
known_reference(:,end) = 1;

xt_reference_added  = [xt known_reference];

% Add reference
% % without separation
% xt_zeropad_reference_added(10:55,42:45) = 1;
% xt_zeropad_reference_added(13:16,40:42) = 1;

% % with separation
% xt_zeropad_reference_added(10:55,107:110) = 1;
% xt_zeropad_reference_added(13:16,105:107) = 1;

% known support
known_reference_support = zeros(size(xt_reference_added));
known_reference_support(:,size(xt,2)+1:end) = 1;

%% Fourier Measurements

Image_Ro = size(xt_reference_added,1);
Image_Co = size(xt_reference_added,2);
Image_Size = size(xt_reference_added);

Measurement_Type = 'fourier';                                    % 'maskFourier', 'Gaussian-Complex', 'fourier','DCT'
n = numel(xt_zeropad_reference_added);                           % Total number of samples in the original signal
m = 4*n;
MaskPatterns = ones(size(xt_zeropad_reference_added));
Image_Support = ones(size(xt_zeropad_reference_added));

Random_Seed = 1;
[A, At, y] =  buildMeasurementMatrix(xt_reference_added,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
b = abs(y);
opts.xt = xt_reference_added;


%% Components in autocorrelation
Autocorr_xt_reference_added = real(ifft2(reshape(b,2*size(xt_zeropad_reference_added)).^2));

% component 1: autocorrelation  of reference
Autocorr_knownReference = real(ifft2(reshape(abs(fft2(known_reference,2*size(known_reference,1),2*size(known_reference,2))),2*size(known_reference)).^2));

% component 2: autocorrelation of unknown signal
cross_corr_nonlinear = xcorr2(xt_zeropad,xt_zeropad);

% component 3: cross correlation of reference and unknown signal
cross_corr_linear = xcorr2(xt_zeropad,known_reference);

fig900 = figure;
fig900.Position = [100, 200, 1200, 300];
[ha, pos] = tight_subplot(1,4,[.00 .00],[0.07 .07],[0.02,0.02]);

axes(ha(1))
imagesc(xt_zeropad_reference_added);colormap gray;title('Original');xticks([]);yticks([])

axes(ha(2))
imagesc(ifftshift(sqrt(Autocorr_xt_reference_added.*sign(Autocorr_xt_reference_added))));colormap gray;title('Autocorrelation');xticks([]);yticks([])

% axes(ha(3))
% imagesc(ifftshift(sqrt(Autocorr_knownReference.*sign(Autocorr_knownReference))));colormap gray;title(' Reference autocorrelation');xticks([]);yticks([])

axes(ha(3))
imagesc(((sqrt(cross_corr_nonlinear.*sign(cross_corr_nonlinear)))));colormap gray;title('Unknown non-linear part');xticks([]);yticks([])

axes(ha(4))
imagesc(((sqrt(cross_corr_linear.*sign(cross_corr_linear)))));colormap gray;title('Unknown linear part');xticks([]);yticks([])


%% reconstruction using our method
x0 = zeros(n,1);
% x0(known_reference_support ==1)= known_reference(known_reference_support ==1);

opts.xt = xt_zeropad_reference_added;
opts.positivity = 1;
opts.support = 0;
opts.knownReference = 1;
opts.Iters = 1000;
opts.lambda  = 10000;
opts.StepSize = 5e-5;
opts.knownReference_support = known_reference_support;
opts.knownReference_values = xt_zeropad_reference_added(opts.knownReference_support == 1);
[x,measurement_error] = PRGradientDescentSolver(x0,A,At,b,opts);
x = reshape(x,size(xt_zeropad_reference_added));
x = x(:,1:75);

%% Reconstruction based on linear inverse problems candes/heraldo y = R*X
Autocorr_diff = diff(Autocorr_xt_reference_added,1,1);
autocorr_diff_centered = ifftshift(Autocorr_diff);
x_recovered_heraldo = autocorr_diff_centered(1:50,1:75);

Autocorr_xt_reference_added_centered = ifftshift(Autocorr_xt_reference_added);
cross_corr_est = Autocorr_xt_reference_added_centered(:,1:75);
x_recovered_candes = diff(cross_corr_est,1,1);
x_recovered_candes = x_recovered_candes(1:50,1:end-3);

%% Plot separately

fig400 = figure;
fig400.Position = [100, 200, 400, 300];
[hc_1, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
% original
axes(hc_1);imagesc(xt_zeropad_reference_added);colormap(hc_1,'gray');xticks([]); yticks([])

% autocorrelation
fig500 = figure;
fig500.Position = [100, 200, 400, 300];
[hc_2, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
imagesc(ifftshift(sqrt(Autocorr_xt_reference_added.*sign(Autocorr_xt_reference_added))));colormap gray;xticks([]);yticks([])

% autocorrelation  of reference
fig600 = figure;
fig600.Position = [100, 200, 400, 300];
[hc_3, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_3);imagesc(ifftshift(sqrt(Autocorr_knownReference.*sign(Autocorr_knownReference))));colormap(hc_3,'gray');xticks([]); yticks([])

% autocorrelation of unknown signal
fig700 = figure;
fig700.Position = [100, 200, 400, 300];
[hc_4, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_4);imagesc(sqrt(cross_corr_nonlinear.*sign(cross_corr_nonlinear)));colormap(hc_4,'gray');xticks([]); yticks([])

% cross correlation of reference and unknown signal
fig800 = figure;
fig800.Position = [100, 200, 400, 300];
[hc_5, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_5);imagesc(sqrt(cross_corr_linear.*sign(cross_corr_linear)));colormap(hc_5,'gray');xticks([]); yticks([])

% Reconstruction using our method
fig900 = figure;
fig900.Position = [100, 200, 400, 300];
[hc_6, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_6);imagesc(x);colormap(hc_6,'gray');xticks([]); yticks([])

% Reconstruction using linear inverse problems: derivative of
% autocorrelation
fig1000 = figure;
fig1000.Position = [100, 200, 400, 300];
[hc_7, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_7);imagesc(ifftshift(sqrt(Autocorr_diff.*sign(Autocorr_diff))));colormap(hc_7,'gray');xticks([]); yticks([])

% Reconstruction using linear inverse problems: candes 
fig11 = figure;
fig11.Position = [100, 200, 400, 300];
[hc_8, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_8);imagesc(x_recovered_candes);colormap(hc_8,'gray');xticks([]); yticks([])

