clc
close all
clear variables

frame1 = im2double(imread('coins.png'));
frame1_ref = imref2d(size(frame1));

moving_mnist_all = readNPY('mnist_test_seq.npy');

known_refrence = zeros(64,64);
% known_refrence(1:64,1:2) = 1;
% % known_refrence(1:64,63:64) = 1;
% known_refrence(1:2,1:64) = 1;
% % known_refrence(63:64,1:64) = 1;

known_refrence(10:end,10:13) = 1;
known_refrence(10:13,10:end) = 1;



%% Phase Retrieval
% Constraint
opts.positivity = 1;
opts.support = 0;
opts.Iters = 500;
opts.StepSize = 5e-5;
opts.lambda = 10000;
num_frames = 10;

num_trials = 1;
rng(0)

for v = 1: 1
    v
    sample_video = im2double(squeeze(moving_mnist_all(:,v,:,:)));
    sample_video = permute(sample_video,[2 3 1]);
    
    for f = 1:num_frames
        f
        tic
        xt = squeeze(sample_video(:,:,f));
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
        
        opts.knownReference = 0;
        for tr = 1:num_trials
            x0 = randn(size(xt));
            [x_temp,measurement_err] = PRGradientDescentSolver(x0,A,At,b,opts);
            x_regularPR(v,f,tr,:,:) = reshape(x_temp,size(xt));
            psnr_regularPR(v,f,tr) = psnr(normalize(xt(:)),normalize(x_temp));
            ssim_regularPR(v,f,tr) = ssim(normalize(xt(:)),normalize(x_temp));
            
        end
        
        % add reference: no info
        
        xt_ref_added = xt + known_refrence;
        
        max_xt = max(xt(:));
        
        xt_ref_added((xt_ref_added> max_xt)) = max_xt;
        
        [A, At, y] = buildMeasurementMatrix(xt_ref_added,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
        b = abs(y);
        opts.xt = xt_ref_added;
        
        opts.knownReference = 0;
        x0 = zeros(size(xt_ref_added));
        [x_temp,measurement_err] = PRGradientDescentSolver(x0,A,At,b,opts);
        x_ref_added_PR_NoSideInfo(v,f,:,:) = reshape(x_temp,size(xt_ref_added));
        
        psnr_ref_added_PR_NoSideInfo(v,f) = psnr(normalize(xt_ref_added(:)),normalize(x_temp));
        ssim_ref_added_PR_NoSideInfo(v,f) = ssim(normalize(xt_ref_added(:)),normalize(x_temp));
        
        % add reference: use info
        opts.knownReference = 1;
        opts.knownReference_support = known_refrence;
        opts.knownReference_values = xt_ref_added(opts.knownReference_support == 1);
        
        x0 = zeros(size(xt_ref_added));
        [x_temp,measurement_err] = PRGradientDescentSolver(x0,A,At,b,opts);
        x_ref_added_PR_withSideInfo(v,f,:,:) = reshape(x_temp,size(xt_ref_added));
        psnr_ref_added_PR_withSideInfo(v,f) = psnr(normalize(xt_ref_added(:)),normalize(x_temp));
        ssim_ref_added_PR_withSideInfo(v,f) = ssim(normalize(xt_ref_added(:)),normalize(x_temp));
        
        x_original(v,f,:,:) = xt;
        time = toc
    end
end


%%
num_frames = 5;

fig = figure(300); fig.Position = [100 250 1000 700];
[ha, pos] = tight_subplot(4, num_frames, [0.01 0.01], [0.01 0.01], [0.04 0.02]);

v = 1;
tr = 1;

k = 1;

for f = 1:num_frames
    axes(ha(k))
    imagesc(squeeze(x_original(v,f,:,:)));colormap gray; xticks([]);yticks([])
    if(k ==1)
        ylabel('Original','FontSize',22)
    end
    
    axes(ha(k+num_frames))
    imagesc(squeeze(x_regularPR(v,f,tr,:,:)));colormap gray; xticks([]);yticks([])
    if(k ==1)
        ylabel('Reg PR','FontSize',22)
    end
    
    axes(ha(k+2*num_frames))
    imagesc(squeeze(x_ref_added_PR_NoSideInfo(v,f,:,:)));colormap gray; xticks([]);yticks([])
    
    if(k ==1)
        ylabel('Test 1','FontSize',22)
    end
    
    axes(ha(k+3*num_frames))
    imagesc(squeeze(x_ref_added_PR_withSideInfo(v,f,:,:)));colormap gray; xticks([]);yticks([])
    
    if(k ==1)
        ylabel('Test 2','FontSize',22)
    end
    k = k +1
end



%% figure for paper
fig = figure(1); fig.Position = [100 250 1000 150];
[ha, pos] = tight_subplot(1, num_frames, [0.00 0.00], [0.0 0.0], [0.0 0.0]);

v = 1;
tr = 1;

k = 1;

for f = 3:num_frames+2
    axes(ha(k))
    imagesc(squeeze(x_original(v,f,:,:)));colormap gray; xticks([]);yticks([])
    k = k +1;
end

fig = figure(2); fig.Position = [100 250 1000 150];
[hb, pos] = tight_subplot(1, num_frames, [0.00 0.00], [0.0 0.0], [0.0 0.0]);

v = 1;
tr = 10;

k = 1;

for f = 3:num_frames+2
    axes(hb(k))
    imagesc(squeeze(x_ref_added_PR_NoSideInfo(v,f,:,:)));colormap gray; xticks([]);yticks([])
    k = k +1;
end



fig = figure(3); fig.Position = [100 250 1000 150];
[hc, pos] = tight_subplot(1, num_frames, [0.00 0.00], [0.0 0.0], [0.0 0.0]);

v = 1;
tr = 1;

k = 1;

for f = 3:num_frames+2
    axes(hc(k))
    imagesc(squeeze(x_regularPR(v,f,tr,:,:)));colormap gray; xticks([]);yticks([])    
    k = k +1;
end

fig = figure(4); fig.Position = [100 250 1000 150];
[hd, pos] = tight_subplot(1, num_frames, [0.0 0.0], [0.0 0.0], [0.00 0.00]);


v = 1;
tr = 1;

k = 1;

for f = 3:num_frames+2
    axes(hd(k))
    imagesc(squeeze(x_ref_added_PR_withSideInfo(v,f,:,:)));colormap gray; xticks([]);yticks([])    
    k = k +1;
end



% save('moving_mnist_corner','x_all_original','x_all','x_all_regularPR','psnr_all')