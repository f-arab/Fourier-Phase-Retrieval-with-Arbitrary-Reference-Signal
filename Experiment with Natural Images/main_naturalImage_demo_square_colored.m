clc
close all
clear variables

%% Signal
colored_image_name = 'baboon.png';
image_name = {'cameraman.tif','coins.png'};

% Phase Retrieval Parameters

% Constraint
opts.positivity = 1;
opts.support = 0;
opts.knownReference = 1;
opts.Iters = 1000 ;
opts.StepSize = 5e-5;
opts.lambda = 5000;

% Colored Image
colored_image = im2double((imread(colored_image_name)));
colored_image = imresize(colored_image,[100,100]);
im_row = size(colored_image,1);
im_col = size(colored_image,2);
n = im_row * im_col;
num_channels = size(colored_image,3);

known_percent = [49 36 25 16 9 4 1];
square_length = sqrt(known_percent/100*im_row*im_col);
center = 100/2;
for k = 1:length(known_percent)+1
    clc,k    
    for ch = 1:num_channels
        ch
        xt = colored_image(:,:,ch);
        xt = squeeze(xt);
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
        
        
        if (k == (length(known_percent)+1))
            opts.knownReference = 0;
        end
        
        
        if (opts.knownReference == 1)
            opts.knownReference_support = zeros(size(xt));
            opts.knownReference_support(center- square_length(k)/2:center+ square_length(k)/2,center- square_length(k)/2:center+ square_length(k)/2) = 1;
            
            opts.knownReference_values = xt(opts.knownReference_support == 1);
        end
        
        x0 = zeros(size(xt));
        
        [x_all(k,ch,:),measurement_err(k,ch,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
        x_all_colored(k,:,:,ch) = reshape(squeeze(x_all(k,ch,:)),100,100);
    end
    psnr_all_colored(k) = psnr(squeeze(x_all_colored(k,:,:,:)),colored_image);
    ssim_all_colored(k) = ssim(squeeze(x_all_colored(k,:,:,:)),colored_image);
    
end

%% compute PSNR and SSIM for unknown part

for k = 1:length(square_length)
    
    opts.knownReference_support = zeros(size(xt));
    opts.knownReference_support(center- square_length(k)/2:center+ square_length(k)/2,center- square_length(k)/2:center+ square_length(k)/2) = 1;
            
     clear colored_image_unknown x_all_colored_unknown
     for ch = 1:3
     colored_image_temp = colored_image(:,:,ch);
     colored_image_unknown(:,:,ch) = colored_image_temp(opts.knownReference_support ==0);
     
     x_all_temp = x_all_colored(k,:,:,ch);
     x_all_colored_unknown(:,:,ch) = x_all_temp(opts.knownReference_support ==0);
     end
     
     ssim_all_colored(k) = ssim(squeeze(x_all_colored(k,:,:,:)),colored_image);

     psnr_unknown_colored(k) = psnr(x_all_colored_unknown,colored_image_unknown);
     ssim_unknown_colored(k) = ssim(x_all_colored_unknown,colored_image_unknown);

end


%%
fig = figure(300); fig.Position = [100 250 1100 680];
[ha, pos] = tight_subplot(3, 6, [0.1 0.01], [0.1 0.05], [0.01 0.01]);

for v = 1:5
    axes(ha(v))
    
    if (v ~= 1)
        imagesc(squeeze(x_all_colored(v-1,:,:,:)))
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        title(sprintf('%d%%',known_percent(v-1)),'FontSize',20)
        %         xlabel({sprintf('(%0.2f, %0.2f)',ssim_all_colored(v-1),psnr_all_colored(v-1)),sprintf('(%0.2f, %0.2f)',ssim_unknown_colored(v-1),psnr_unknown_colored(v-1))},'FontSize',18)
        xlabel({strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim_all_colored(v-1),psnr_all_colored(v-1))),strcat('\color{blue}',sprintf('(%0.2f, %0.2f)',ssim_unknown_colored(v-1),psnr_unknown_colored(v-1)))},'FontSize',18)
        
    end
    
    if (v == 1)
        imagesc(colored_image)
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        title('Original','FontSize',20)
        xlabel('(SSIM, PSNR)','FontSize',18)
    end
   
    
end
axes(ha(6))
imagesc(squeeze(x_all_colored(7,:,:,:)))
axes(ha(6))
set(gca,'xtick',[])
set(gca,'ytick',[])
title('No side Info','FontSize',20)
xlabel(strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim_all_colored(7),psnr_all_colored(7))),'FontSize',18)


%% Grayscale images
image_name = {'cameraman.tif' 'coins.png'};
for im = 1: length(image_name)
    opts.knownReference = 1;
    clc,im
    objects = double((imread(image_name{im})));
    objects = imresize(objects,[100,100]);
    xt = objects;
    
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
    x_est(im,1,:) = xt(:);
    
    for k = 1:length(known_percent)+1
        
         if (k == (length(known_percent)+1))
            opts.knownReference = 0;
        end
        
        if (opts.knownReference == 1)
            opts.knownReference_support = zeros(size(xt));
            opts.knownReference_support(center- square_length(k)/2:center+ square_length(k)/2,center- square_length(k)/2:center+ square_length(k)/2) = 1;
            
            opts.knownReference_values = xt(opts.knownReference_support == 1);
        end
        
        x0 = zeros(size(xt));
        [x_est(im,k+1,:),measurement_err_2(im,k,:)] = PRGradientDescentSolver(x0,A,At,b,opts);
        
        psnr_all(im,k) = psnr(normalize(x_est(im,k+1,:)),normalize(x_est(im,1,:)));
        ssim_all(im,k) = ssim(x_est(im,k+1,:),x_est(im,1,:));
        
        
    end
end

%% Compute PSNR and SSIM for unknown part

for k = 1:length(square_length)
    
    opts.knownReference_support = zeros(size(xt));
    opts.knownReference_support(center- square_length(k)/2:center+ square_length(k)/2,center- square_length(k)/2:center+ square_length(k)/2) = 1;
    
    
    psnr_all(1,k) = psnr(normalize(x_est(1,k+1,:)),normalize(x_est(1,1,:)));
    ssim_all(1,k) = ssim(normalize(x_est(1,k+1,:)),normalize(x_est(1,1,:)));
    
    
    psnr_all(2,k) = psnr(normalize(x_est(2,k+1,:)),normalize(x_est(2,1,:)));
    ssim_all(2,k) = ssim(normalize(x_est(2,k+1,:)),normalize(x_est(2,1,:)));
    
    
     x_original_temp_camera = x_est(1,1,opts.knownReference_support == 0 );
     x_original_temp_coins = x_est(2,1,opts.knownReference_support == 0 );
     
     x_est_temp_camera = x_est(1,k+1,opts.knownReference_support == 0 );
     x_est_temp_coins = x_est(2,k+1,opts.knownReference_support == 0 );

     psnr_unknown(1,k) = psnr(normalize(x_est_temp_camera),normalize(x_original_temp_camera));
     ssim_unknown(1,k) = ssim(normalize(x_est_temp_camera),normalize(x_original_temp_camera));
     
     psnr_unknown(2,k) = psnr(normalize(x_est_temp_coins),normalize(x_original_temp_coins));
     ssim_unknown(2,k) = ssim(normalize(x_est_temp_coins),normalize(x_original_temp_coins));

end


%% cameraman
k = 7;
for v = 1:5
    axes(ha(k))
    imagesc(reshape(x_est(1,v,:),size(xt)))
    colormap gray
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    if (v~=1)
        xlabel({strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim_all(1,v-1),psnr_all(1,v-1))),strcat('\color{blue}',sprintf('(%0.2f, %0.2f)',ssim_unknown(1,v-1),psnr_unknown(1,v-1)))},'FontSize',18)
    end
    k = k +1 ;
end

axes(ha(12))
imagesc(reshape(x_est(1,end,:),size(xt)))
colormap gray
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel(strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim(normalize(x_est(1,1,:)),normalize(x_est(1,end,:))),psnr(normalize(x_est(1,1,:)),normalize(x_est(1,end,:))))),'FontSize',18)

    
%% coins
k = 13;
for v = 1:5
    axes(ha(k))
    imagesc(reshape(x_est(2,v,:),size(xt)))
    colormap gray
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    if (v~=1)
        xlabel({strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim_all(2,v-1),psnr_all(2,v-1))),strcat('\color{blue}',sprintf('(%0.2f, %0.2f)',ssim_unknown(2,v-1),psnr_unknown(2,v-1)))},'FontSize',18)
    end
    
    k = k +1 ;
end

axes(ha(18))
imagesc(reshape(x_est(2,end,:),size(xt)))
colormap gray
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel(strcat('\color[rgb]{0 .5 0}',sprintf('(%0.2f, %0.2f)',ssim(normalize(x_est(2,1,:)),normalize(x_est(2,end,:))),psnr(normalize(x_est(2,1,:)),normalize(x_est(2,end,:))))),'FontSize',18)
% save('natural_images_coloredBaboon_notSaturated_square','x_est','psnr_all','ssim_all','x_all_colored', 'psnr_all_colored','known_percent','psnr_unknown_pixels','psnr_known_pixels')


%% plot psnr and ssim
ssim_all_gray_colored = [ssim_all_colored;ssim_all];
% psnr_all_gray_colored = [psnr_all_colored;psnr_all];

figure
plot(known_percent,ssim_all_gray_colored(1,1:end),'r-*','Linewidth',2.5);grid on; grid minor;hold on
plot(known_percent,ssim_all_gray_colored(2,1:end),'b-o','Linewidth',2.5);hold on
plot(known_percent,ssim_all_gray_colored(3,1:end),'g-^','Linewidth',2.5)
ylabel('SSIM','FontSize',16);xlabel('Known Area (%)','FontSize',16)
legend('Baboon', 'Cameraman', 'Coins','FontSize',16)

% 
% figure
% plot(known_percent,psnr_all_gray_colored(1,1:end-1),'r-*','Linewidth',2.5);grid on; grid minor;hold on
% plot(known_percent,psnr_all_gray_colored(2,1:end-1),'b-o','Linewidth',2.5);hold on
% plot(known_percent,psnr_all_gray_colored(3,1:end-1),'g-^','Linewidth',2.5)
% ylabel('PSNR','FontSize',16);xlabel('Known Area (%)','FontSize',16)
% legend('Baboon', 'Cameraman', 'Coins','FontSize',16)

