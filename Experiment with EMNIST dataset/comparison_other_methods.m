clc
close all
clear variables

%% Signal
addpath(genpath('data/EMNIST'))

dataset_letters = load('emnist-letters');
dataset_letters_images = dataset_letters.dataset.train.images;
total_chars = size(dataset_letters_images,1);

rng(59)
x1_idx = randi(total_chars,1);
x2_idx = randi(total_chars,1);

x1 = im2double(reshape(dataset_letters_images(x1_idx,:),28,28));
x2 = im2double(reshape(dataset_letters_images(x2_idx,:),28,28));

xt = [x1 x2];
%% First Scenario: All the methods work
xt_zeropad = zeros(2*size(xt));
xt_zeropad(15:14+size(xt,1),15:14+size(xt,2)) = xt;

known_reference_support = zeros(size(xt_zeropad));
% %25:23
% known_reference_support(27:29,1:end-20) = 1;
% known_reference_support(26:30,end-23:end-20) = 1;

% known_reference_support(1:end-10,40:43) = 1;
% known_reference_support(end-23:end-20,37:45) = 1;


known_reference_support(10:55,42:45) = 1;
known_reference_support(13:16,40:42) = 1;


xt_zeropad_reference_added = xt_zeropad;
xt_zeropad_reference_added(known_reference_support == 1) = 1;

fig200 = figure;
fig200.Position = [100, 200, 1290, 250];
[ha, pos] = tight_subplot(1,3,[.03 .03],[0.03 .16],[0.01,0.01]);
% axes(ha(1));imagesc(xt_zeropad_reference_added);colormap(ha(1),'gray');xticks([]); yticks([]);title('Original Image ','FontSize',20)

Image_Ro = size(xt_zeropad_reference_added,1);
Image_Co = size(xt_zeropad_reference_added,2);
Image_Size = size(xt_zeropad_reference_added);

%% Fourier Measurements
Measurement_Type = 'fourier';                                    % 'maskFourier', 'Gaussian-Complex', 'fourier','DCT'
n = numel(xt_zeropad_reference_added);                           % Total number of samples in the original signal
m = 4*n;
MaskPatterns = ones(size(xt_zeropad_reference_added));
Image_Support = ones(size(xt_zeropad_reference_added));

Random_Seed = 1;
[A, At, y] =  buildMeasurementMatrix(xt_zeropad_reference_added,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
b = abs(y);
opts.xt_zeropad_reference_added = xt_zeropad_reference_added;

%% Heraldo Method
Autocorr_x_zeropad_reference_added = real(ifft2(reshape(b,2*size(xt_zeropad_reference_added)).^2));

autocorr_diff = diff(Autocorr_x_zeropad_reference_added,1,1);
autocorr_diff_centered = ifftshift(autocorr_diff);
x_recovered_heraldo = autocorr_diff_centered(1:50,1:75);

% axes(ha(2));imagesc((imadjust(autocorr_diff_centered)));colormap(ha(2),'gray');xticks([]); yticks([]);title('Derivative of autocorrelation ','FontSize',20)
% axes(ha(3));imagesc((x_recovered_heraldo));colormap(ha(3),'gray');xticks([]); yticks([]);title('Recovered (Heraldo) ','FontSize',20)

%% Candes method
Autocorr_x_zeropad_reference_added_cenered = ifftshift(Autocorr_x_zeropad_reference_added);
cross_corr = Autocorr_x_zeropad_reference_added_cenered(:,1:75);

fig300 = figure;
fig300.Position = [100, 200, 1290, 250];
[hb, pos] = tight_subplot(1,3,[.03 .03],[0.03 .16],[0.01,0.01]);
% axes(hb(1));imagesc(ifftshift(Autocorr_x_zeropad_reference_added));colormap(hb(1),'jet');xticks([]); yticks([]);title('Autocorrelation ','FontSize',20)
% axes(hb(2));imagesc(cross_corr);colormap(hb(2),'jet');title('Cross-correlation (linear)','FontSize',20);xticks([]); yticks([])

x_recovered_candes = diff(cross_corr,1,1);
x_recovered_candes = x_recovered_candes(1:50,1:end-3);
% axes(hb(3));imagesc(x_recovered_candes);title('Recovered (Candes)','FontSize',20);colormap(hb(3),'gray');xticks([]); yticks([])


%% our method

x0 = zeros(n,1);
% x0(known_reference_support == 1) = xt_zeropad_reference_added(known_reference_support == 1);

opts.xt = xt_zeropad_reference_added;
opts.positivity = 1;
opts.support = 0;
opts.knownReference = 1;
opts.Iters = 10;
opts.lambda  = 10000;
opts.StepSize = 5e-5;
opts.knownReference_support = known_reference_support;
opts.knownReference_values = xt_zeropad_reference_added(opts.knownReference_support == 1);
[x,measurement_error] = PRGradientDescentSolver(x0,A,At,b,opts);
x = reshape(x,size(xt_zeropad_reference_added));
%% Plot all in one figure

fig400 = figure;
fig400.Position = [100, 200, 400, 300];
[hc_1, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
% original
axes(hc_1);imagesc(xt_zeropad_reference_added);colormap(hc_1,'gray');xticks([]); yticks([])

% autocorrelation
fig500 = figure;
fig500.Position = [100, 200, 400, 300];
[hc_2, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_2);imagesc(sqrt(ifftshift(Autocorr_x_zeropad_reference_added.*sign(Autocorr_x_zeropad_reference_added))));colormap(hc_2,'gray');xticks([]); yticks([])

% derivative of autocorrelation
fig600 = figure;
fig600.Position = [100, 200, 400, 300];
[hc_3, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_3);imagesc((sqrt(autocorr_diff_centered.*sign(autocorr_diff_centered))));colormap(hc_3,'gray');xticks([]); yticks([])

% proposed method
fig700 = figure;
fig700.Position = [100, 200, 400, 300];
[hc_4, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_4);imagesc(x);colormap(hc_4,'gray');xticks([]); yticks([])

% candes method: linear inverse problem
fig800 = figure;
fig800.Position = [100, 200, 400, 300];
[hc_5, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_5);imagesc(x_recovered_candes);colormap(hc_5,'gray');xticks([]); yticks([])


% heraldo method
fig900 = figure;
fig900.Position = [100, 200, 400, 300];
[hc_6, pos] = tight_subplot(1,1,[.00 .00],[0.00 .00],[0.00,0.00]);
axes(hc_6);imagesc((x_recovered_heraldo));colormap(hc_6,'gray');xticks([]); yticks([])


