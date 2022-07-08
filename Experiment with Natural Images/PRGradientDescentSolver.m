function [x,measurement_error] = PRGradientDescentSolver(x0,A,At,b,opt)
tic
x = x0(:);
x_new = x;
Image_Size = size(opt.xt);

    function y = Afun(x,transp_flag)
        if strcmp(transp_flag,'transp')       % y = A'*x
            y = At(x);
        elseif strcmp(transp_flag,'notransp') % y = A*x
            y = A(x);
        end
    end

    function x = temp(x_r,knownReference_support,n)
        x = zeros(n,1);
        x(knownReference_support) = x_r;
    end
if (opt.knownReference)
    xr = opt.knownReference_values;
    
    B = @(x) x(opt.knownReference_support==1);
    Bt = @(xr) temp(xr,(opt.knownReference_support==1),length(x));
end

for k = 1: opt.Iters
    
    % Performance Evaluation
    measurement_error(k) = norm(b-abs(A(x)),2)/norm(b,2);
    if (opt.knownReference)
        known_ref_loss(k) = norm(opt.knownReference_values - B(x),2)/norm(opt.knownReference_values,2);
    end

    
%     x_psnr(k) = psnr((opt.xt),(reshape(x,Image_Size)));
% % %     
%     h1 = figure(100);h1.Position = [100,100,900,250]; subplot(1,3,1);plot(measurement_error,'b','Linewidth',2.5);grid on; grid minor;xlabel('Iteration');title('Measurement Error')
% %     subplot(1,3,2);plot(known_ref_loss,'r','Linewidth',2.5);grid on; grid minor;xlabel('Iteration');title('Regularizer Loss')
%     
%     figure(100); subplot(1,3,3);plot(x_psnr,'Linewidth',2.5);grid on; grid minor;xlabel('Iteration');title('Reconstruction Performance (PSNR)')
%     h2 = figure(200);h2.Position = [400,400,1000,300]; subplot(1,2,1);imagesc(opt.xt);title('Original');colorbar;subplot(1,2,2);imagesc(reshape(real(x),Image_Size));title('Recovered');colorbar ;colormap gray
%     drawnow

    % Update Phase
    p = exp(1i*phase(A(x)));
    
    l_gradient = At(b.*p-A(x));
    
    if (opt.knownReference)
        l_gradient = At(b.*p-A(x)) + opt.lambda*Bt(xr - B(x));
    end
    x_p = x + opt.StepSize*l_gradient;
    
    if (opt.positivity)
        inds = x_p<0;  % Get indices that are outside the non-negative constraints
        inds2 = ~inds; % Get the complementary indices
        
        % hybrid input-output (see Section V, Equation (44))
        x_new(inds) = x(inds) - 0.5*x_p(inds);
        x_new(inds2) = x_p(inds2);
    else % Otherwise, its update is the 7 same as the GerchBerg-Saxton algorithm
        x_new = x_p;
    end
    x = real(x_new);
    
    if (opt.support)
        x(opt.objects_support == 0) = 0;
    end
end
end
