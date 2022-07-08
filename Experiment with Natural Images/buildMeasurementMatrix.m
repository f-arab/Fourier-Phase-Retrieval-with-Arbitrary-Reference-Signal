function  [A, At, y] =  buildMeasurementMatrix(xt,Image_Support,MaskPatterns,Measurement_Type,M,Random_Seed)

Image_Dim = size(xt);

%% Make linear operators that act on a vectorized image

switch Measurement_Type
    case  'fourier'
        A = @(x) fourierMeasurementOperator(x, Image_Support,Image_Dim,M);
        At = @(y) fourierTransposeOperator(y, Image_Support, Image_Dim,M);
        
    case 'maskFourier'
        A = @(x) codedFourierMeasurementOperator(x, Image_Support,MaskPatterns,Image_Dim,M);
        At = @(y) codedFourierTransposeOperator(y, Image_Support,MaskPatterns, Image_Dim,M);
        
    case 'Gaussian-Complex'
        rng(Random_Seed)
        A_matrix  = randn(M, length(xt(:)))+1i*randn(M,length(xt(:)));
        
        A = @(x) A_matrix*x;
        At = @(y) A_matrix'*y;
        
    case 'Gaussian-Real'
        rng(Random_Seed)
        A_matrix  = randn(M,length(xt(:)));
        %         [U S V] = svd(A_matrix);
        %         N = min(M,length(xt(:)));
        %         S(1:N,1:N) = diag(linspace(10,0.001,N));
        %         A_matrix = U*S*V';
        
        A = @(x) A_matrix*x;
        At = @(y) A_matrix'*y;
    case  'DCT'
        A = @(x) DCTMeasurementOperator(x, Image_Support,Image_Dim,M);
        At = @(y) DCTTransposeOperator(y, Image_Support, Image_Dim,M);
end

y = A(xt(:));

end

% 'fourier'
function y = fourierMeasurementOperator(x, support,dims,m)

x = reshape(x, dims);   % image comes in as a vector.  Reshape to rectangle
% Compute measurements
y = fft2(support.*x,sqrt(m/prod(dims))*dims(1),sqrt(m/prod(dims))*dims(2));
y = y(:);
end

function x = fourierTransposeOperator(y, support,dims,m)
n1 = dims(1);
n2 = dims(2);
y = reshape(y, sqrt(m/prod(dims))*dims(1),sqrt(m/prod(dims))*dims(2));   % image comes in as a vector.  Reshape to rectangle

x = n1*n1*ifft2(y);
x = x(1:n1,1:n2).*conj(support);
x = x(:);
end


% 'maskFourier'
function y = codedFourierMeasurementOperator(x, support,maskPatterns, dims,m)

x = reshape(x, dims);   % image comes in as a vector.  Reshape to rectangle
[n1,n2] = size(x);
L = size(maskPatterns,3);              % get number of masks

% Compute measurements
copies = repmat(x,[1,1,L]);
y = fft2(maskPatterns.*copies);
y = y(:);
end

function x = codedFourierTransposeOperator(y, support,maskPatterns, dims,m)

n1 = dims(1);
n2 = dims(2);
L = size(maskPatterns,3);              % get number of masks
y = reshape(y, [n1,n2,L]);   % image comes in as a vector.  Reshape to rectangle

x = n1*n2*ifft2(y).*conj(maskPatterns);
x = sum(x,3);
x = x(:);
end


% 'DCT'
function y = DCTMeasurementOperator(x, support,dims,m)

x = reshape(x, dims);   % image comes in as a vector.  Reshape to rectangle
% Compute measurements
y = dct2(support.*x,sqrt(m),sqrt(m));
y = y(:);
end


function x = DCTTransposeOperator(y, support,dims,m)
n1 = dims(1);
n2 = dims(2);
y = reshape(y, sqrt(m),sqrt(m));   % image comes in as a vector.  Reshape to rectangle

x = m*idct2(y);
x = x(1:n1,1:n2).*conj(support);
x = x(:);
end
