clc
close all
clear variables

frame1 = im2double(imread('coins.png'));
frame1_ref = imref2d(size(frame1));

moving_mnist_all = readNPY('mnist_test_seq.npy');

known_refrence = zeros(64,64);
known_refrence(1:64,1:2) = 1;
% known_refrence(1:64,63:64) = 1;
known_refrence(1:2,1:64) = 1;
% known_refrence(63:64,1:64) = 1;


%% Phase Retrieval
% Constraint
opts.positivity = 1;
opts.support = 0;
opts.knownReference = 1;
opts.Iters = 500;
opts.StepSize = 5e-5;
opts.lambda = 10000;


for v = 1: 10
    v
    sample_video = im2double(squeeze(moving_mnist_all(:,v,:,:)));
    sample_video = permute(sample_video,[2 3 1]);
    
    sample_video_refernce_added = sample_video + repmat(known_refrence,1,1,20);
    
    for f = 1:size(sample_video_refernce_added,3)
        f
        tic
        xt = squeeze(sample_video_refernce_added(:,:,f));
        Image_Ro = size(xt,1);
        Image_Co = size(xt,2);
        Image_Size = size(xt);
        
        %% Phase Retrieval
        Measurement_Type = 'fourier';            % 'maskFourier', 'Gaussian-Complex', 'fourier','DCT'
        n = numel(xt);                           % Total number of samples in the original signal
        m = 4*n;
        MaskPatterns = ones(size(xt));
        Image_Support = ones(size(xt));
        opts.objects_support = Image_Support;
        
        Random_Seed = 1;
        [A, At, y] = buildMeasurementMatrix(xt,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
        b = abs(y);
        opts.xt = xt;
        
        opts.knownReference_support = known_refrence;
        opts.knownReference_values = xt(opts.knownReference_support == 1);
        
        x0 = zeros(size(xt));
        [x_est,measurement_err(v,f,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
        
        opts.knownReference = 0;
        [x_est_regularPR,measurement_err_regularPR(v,f,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
        
        x_all(:,:,v,f) = reshape(x_est,size(xt));
        x_all_original(:,:,v,f) = xt;
        x_all_regularPR(:,:,v,f) = reshape(x_est_regularPR,size(xt));

        opts.knownReference = 1;
        
        psnr_all(v,f) = psnr(xt,reshape(x_est,size(xt)));
        ssim_all(v,f) = ssim(xt,reshape(x_est,size(xt)));
        t = toc
    end
end


%% PSNR for regular PR with added corner

for v = 1:10
    for f  = 1:20
        psnr_all_regular_added_ref(v,f) = psnr(x_all_regularPR(:,:,v,f),x_all_original(:,:,v,f));
        ssim_all_regular_added_ref(v,f) = ssim(x_all_regularPR(:,:,v,f),x_all_original(:,:,v,f));

    end
end
    
% save('moving_mnist_corner_paper','x_all_original','x_all_regularPR','x_all','psnr_all_regular_added_ref','ssim_all_regular_added_ref','psnr_all','ssim_all')
%% Regular PR for original images without adding corner part
opts.knownReference = 0;
rng(0)

for v = 1: 10
    v,tic
    sample_video = im2double(squeeze(moving_mnist_all(:,v,:,:)));
    sample_video = permute(sample_video,[2 3 1]);
    
    sample_video_refernce_added = sample_video;
    
    for f = 1:size(sample_video_refernce_added,3)
        f
        tic
        xt = squeeze(sample_video_refernce_added(:,:,f));
        Image_Ro = size(xt,1);
        Image_Co = size(xt,2);
        Image_Size = size(xt);
        
        % Phase Retrieval
        Measurement_Type = 'fourier';            % 'maskFourier', 'Gaussian-Complex', 'fourier','DCT'
        n = numel(xt);                           % Total number of samples in the original signal
        m = 4*n;
        MaskPatterns = ones(size(xt));
        Image_Support = ones(size(xt));
        opts.objects_support = Image_Support;
        
        Random_Seed = 1;
        [A, At, y] = buildMeasurementMatrix(xt,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
        b = abs(y);
        opts.xt = xt;
        for tr = 1:1
            x0 = zeros(size(xt));
            [x_est,measurement_err_noInfo(v,f,tr,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
            
            x_all_regularPR_OriginalImage(:,:,v,f,tr) = reshape(x_est,size(xt));
            
            psnr_all_regularPR_original_image(v,f,tr) = psnr(xt,reshape(x_est,size(xt)));
            ssim_all_regularPR_original_image(v,f,tr) = ssim(xt,reshape(x_est,size(xt)));
        end
    end
    t = toc
end
%%
for v = 1:10
    for f  = 1:20
        psnr_all_regularPR_original_image(v,f) = psnr(x_all_regularPR_OriginalImage(:,:,v,f,1),x_all_original(:,:,v,f));
        ssim_all_regularPR_original_image(v,f) = ssim(x_all_regularPR_OriginalImage(:,:,v,f,1),x_all_original(:,:,v,f));

    end
end
%%
fig = figure(300); fig.Position = [100 250 2000 300];
[ha, pos] = tight_subplot(4, size(x_all,4), [0.05 0.01], [0.05 0.05], [0.01 0.01]);

v = 1;
tr = 1;
for ii = 1:size(x_all,4)
    axes(ha(ii))
    imagesc(x_all_original(:,:,v,ii));colormap gray; xticks([]);yticks([])
    if(ii ==1)
        ylabel('Original')
    end
    
    axes(ha(ii+size(x_all,4)))
    imagesc(x_all_regularPR_OriginalImage(:,:,v,ii,tr));colormap gray; xticks([]);yticks([])
    if(ii ==1)
        ylabel('Regular PR 1')
    end
    
    axes(ha(ii+2*size(x_all,4)))
    imagesc(x_all_regularPR(:,:,v,ii));colormap gray; xticks([]);yticks([])
    
    if(ii ==1)
        ylabel('Regular PR 2')
    end
    
    axes(ha(ii+3*size(x_all,4)))
    imagesc(x_all(:,:,v,ii));colormap gray; xticks([]);yticks([])
    
    if(ii ==1)
        ylabel('With side info')
    end
end

% save('moving_mnist_corner','x_all_original','x_all','x_all_regularPR','psnr_all')