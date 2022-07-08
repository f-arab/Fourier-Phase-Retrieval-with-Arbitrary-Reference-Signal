clc
close all
clear variables

%% Signal
image_name =  {'rice.png' 'cameraman.tif' 'coins.png'};
for im = 1: length(image_name)
    im,
    objects = double((imread(image_name{im})));
    objects = round((objects - min(objects(:)))/max(objects(:)));
    
    objects = imresize(objects,[100,100]);
    xt = objects;
    
    Image_Ro = size(xt,1);
    Image_Co = size(xt,2);
    Image_Size = size(xt);
    
    %% Phase Retrieval
    Measurement_Type = 'fourier';            % 'maskFourier', 'Gaussian-Complex', 'fourier','DCT'
    n = numel(xt);                           % Total number of samples in the original signal
    m = 4*n;
    MaskPatterns = ones(size(xt));
    Image_Support = ones(size(xt));
    
    Random_Seed = 1;
    [A, At, y] =  buildMeasurementMatrix(xt,Image_Support,MaskPatterns,Measurement_Type,m,Random_Seed);
    b = abs(y);
    opts.xt = xt;
    
    % Constraint
    opts.positivity = 1;
    opts.support = 0;
    opts.knownReference = 1;
    opts.Iters = 1000;
    opts.objects_support = Image_Support;
    opts.StepSize = 5e-5;
    opts.lambda = 5000;
    square_length = [70 60 50 40 30 20 10];
        
    rng(10)
    center = 100/2;
    for k = 1:length(square_length)
         k
        tic
        %     opts.knownReference_support = (100-known_percent(k))*n/100:n;
        opts.knownReference_support = zeros(size(xt));
        opts.knownReference_support(center- square_length(k)/2:center+ square_length(k)/2,center- square_length(k)/2:center+ square_length(k)/2) = 1;
        opts.knownReference_values = xt(opts.knownReference_support == 1);
        
        x0 = zeros(size(xt));
%       x0 (opts.knownReference_support == 1)= xt(opts.knownReference_support == 1);
        x_est(im,1,:) = xt(:);
        [x_est(im,k+1,:),measurement_err_2(im,k,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
        psnr_all(im,k) = psnr(x_est(im,1,:),x_est(im,k+1,:));
        ssim_all(im,k) = ssim(x_est(im,1,:),x_est(im,k+1,:));
        t = toc
    end
end

% save('natural_images_removedNormalization_center_square_zeroInitial_lambda_5000','x_est','measurement_err_2','known_percent','psnr_all','ssim_all')
%%
fig = figure(300); fig.Position = [100 250 1700 650];
[ha, pos] = tight_subplot(size(x_est,1), size(x_est,2), [0.05 0.01], [0.05 0.05], [0.01 0.01]);
k = 1;
for h = 1:length(image_name)
    for v = 1:size(x_est,2)
        axes(ha(k))
        imagesc(reshape(x_est(h,v,:),size(xt)))
        colormap gray
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        if (h == 1 & v ~= 1)
            title(sprintf('%d%% known',known_percent(v-1)),'FontSize',14)
        end
        if (v~=1)
            xlabel(sprintf('PSNR = %0.2f',psnr_all(h,v-1)),'FontSize',14)

        end
        
        if (h== 1 & v == 1)
            title('Original Images','FontSize',14)
        end
        
        k = k +1 ;
    end
end

known_percent = square_length.^2/prod(size(xt))*100;

figure
plot(known_percent,ssim_all(1,:),'r-*','Linewidth',2.5);grid on; grid minor;hold on
plot(known_percent,ssim_all(2,:),'b-o','Linewidth',2.5);hold on
plot(known_percent,ssim_all(3,:),'g-^','Linewidth',2.5)
ylabel('SSIM','FontSize',16);xlabel('Known Area (%)','FontSize',16)
legend('grayMandrill', 'cameraman', 'coins','FontSize',16)

figure
plot(known_percent,psnr_all(1,:),'r-*','Linewidth',2.5);grid on; grid minor;hold on
plot(known_percent,psnr_all(2,:),'b-o','Linewidth',2.5);hold on
plot(known_percent,psnr_all(3,:),'g-^','Linewidth',2.5)
ylabel('PSNR (dB)','FontSize',16);xlabel('Known Area (%)','FontSize',16)
legend('grayMandrill', 'cameraman', 'coins','FontSize',16)


