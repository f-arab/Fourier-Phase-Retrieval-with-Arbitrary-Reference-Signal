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
opts.Iters = 10;
opts.StepSize = 5e-5;
opts.lambda = 10000;
num_frames = 20;

num_trials = 1;
rng(0)

fig= figure(100)
fig.Position = [100 200 1500 300];

filename = 'testAnimated.gif';
d = 1 ; 
for v = 1: 10
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
        
        xt_autocorr = real(ifft2(reshape(b.^2,2*size(xt))));
        
        opts.knownReference = 0;
        
        % add reference: no info
        
        xt_ref_added = xt + known_refrence;
        [A, At, y] = buildMeasurementMatrix(xt_ref_added,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
        b = abs(y);
        
        xt_ref_added_autocorr = real(ifft2(reshape(b.^2,2*size(xt_ref_added))));
        
        xt_ref_added_autocorr_diff_1 = diff(xt_ref_added_autocorr,1,1);
        xt_ref_added_autocorr_diff_2 = diff(xt_ref_added_autocorr,1,2);
        
        xt_ref_added_autocorr = diff(xt_ref_added_autocorr,1,1);
        xt_ref_added_autocorr = diff(xt_ref_added_autocorr,1,2);
        
        subplot(1,4,1);imagesc(xt_ref_added);colormap gray; title('Original Image');xticks([]);yticks([])
        subplot(1,4,2);imagesc(sqrt(ifftshift(xt_ref_added_autocorr_diff_1.*sign(xt_ref_added_autocorr_diff_1)))); title('Derivative in direction 1');xticks([]);yticks([])
        subplot(1,4,3);imagesc(sqrt(ifftshift(xt_ref_added_autocorr_diff_2.*sign(xt_ref_added_autocorr_diff_2)))); title('Derivative in direction 2');xticks([]);yticks([])

        subplot(1,4,4);imagesc(sqrt(ifftshift(xt_ref_added_autocorr.*sign(xt_ref_added_autocorr))));colormap gray;title('Derivative in direction 1 and 2');xticks([]);yticks([])
        drawnow
        
       axis tight manual % this ensures that getframe() returns a consistent size

      % Capture the plot as an image 
      frame = getframe(fig); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if d == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
      
        opts.xt = xt_ref_added;
        
        opts.knownReference = 0;
        x0 = zeros(size(xt_ref_added));
        [x_temp,measurement_err] = PRGradientDescentSolver(x0,A,At,b,opts);
        
        % add reference: use info
        opts.knownReference = 1;
        opts.knownReference_support = known_refrence;
        opts.knownReference_values = xt_ref_added(opts.knownReference_support == 1);
        
        x0 = zeros(size(xt_ref_added));
        [x_temp,measurement_err] = PRGradientDescentSolver(x0,A,At,b,opts);
        
        d = d +1 ;
        
    end
end