% %  Chris Metzler.
% % Nov 17 2017
%% ImagingDemo.m reconstructsa sqrt(n)xsqrt(n) SLM pattern from a speckle pattern measured on the detector using the GS algorithm.

k=3;%Choose with dataset to use
clearvars -except k

%% Demonstrate Imaging
switch k
    case 1    
        load './../Coherent_Data/AmpSLM_16x16/A_GS.mat'%Contains the transmission matrix and the residual vector
        load './../Coherent_Data/AmpSLM_16x16/YH_squared_test.mat'
        load './../Coherent_Data/AmpSLM_16x16/XH_test.mat'
        n=16^2;
    case 2
        load './../Coherent_Data/PhaseSLM_40x40/A_GS.mat'%Contains the transmission matrix and the residual vector
        load './../Coherent_Data/PhaseSLM_40x40/YH_squared_test.mat'
        load './../Coherent_Data/PhaseSLM_40x40/XH_test.mat'
        n=40^2;
    case 3
        load './../Coherent_Data/AmpSLM_64x64/A_GS.mat'%Contains the transmission matrix and the residual vector
        load './../Coherent_Data/AmpSLM_64x64/YH_squared_test.mat'
        load './../Coherent_Data/AmpSLM_64x64/XH_test.mat'
        n=64^2;
end


X_true = double(XH_test)'/255;
Y= double(YH_squared_test)'.^.5;

use_GPU=1;
if use_GPU
    A=gpuArray(single(A));
    Y=gpuArray(single(Y));
end
good_inds = find(residual_vector<.4);%Throw out the poorly reconstructed rows
Y = Y(good_inds,:);
A = A(good_inds,:);
GS_iters = 100;
disp('started computations')
A_dag = pinv(A);

t0=tic;
X_recon = GS(Y,A,A_dag,GS_iters);
t1=toc(t0)
if use_GPU
    X_recon=gather(X_recon);
end
X_recon = real(X_recon);%./max(real(X_recon),1);

figure;
for i=1:min(size(X_true,2),5)
    subplot(2,5,i); imshow(reshape(X_recon(:,i),sqrt(n),sqrt(n)),[]);
end

for i=1:min(size(X_true,2),5)
    subplot(2,5,i+5); imshow(reshape(X_true(:,i),sqrt(n),sqrt(n)),[]);
end
