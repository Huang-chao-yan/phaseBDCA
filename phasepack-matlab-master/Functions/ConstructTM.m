% %  Chris Metzler.
% % Nov 17 2017
%% ConstructTM.m reconstructs an mxn transmission matrix using p distinct calibration/training x,y pairs using the GS algorithm

k=1;%Choose which transmission matrix to construct.
clearvars -except k

switch k
    case 1
        load ./../Coherent_Data/AmpSLM_16x16/XH_train.mat
        load ./../Coherent_Data/AmpSLM_16x16/YH_squared_train.mat
        n = 16^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=5e3;%Number of rows to compute in parallel. 
    case 2
        load ./../Coherent_Data/PhaseSLM_40x40/XH_train.mat
        load ./../Coherent_Data/PhaseSLM_40x40/YH_squared_train.mat
        n = 40^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=1e3;%Number of rows to compute in parallel.
    case 3
        load ./../Coherent_Data/AmpSLM_64x64/XH_train.mat
        load ./../Coherent_Data/AmpSLM_64x64/YH_squared_train.mat
        n = 64^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=1e3;%Number of rows to compute in parallel. 
end

alpha = 12;% Oversampling factor
p = alpha*n;% Number of measurements
NumOfRows = 256^2; % This controls how much of the TM is reconstructed 
GS_iters = 200; %How many iterations the GS algorithm is run

%% Demonstrate Calibration (learning the transmission matrix)

Y_H = single(YH_squared_train(1:p,:)).^.5;
X_H = single(XH_train(1:p,:))/255;
use_GPU=true;
if use_GPU
    X_H_computations=gpuArray(X_H);
else
    X_H_computations=X_H;
end
X_dag = pinv(X_H_computations);
A=single(zeros(NumOfRows,n));
residual_vector=zeros(NumOfRows,1);
h=waitbar(0,'Computing transmission matrix...');
t00=tic;
for ind = 1:nParallel:NumOfRows
    waitbar(ind/NumOfRows);
    t0=tic;
    ind_end=min(ind+nParallel-1,NumOfRows);
    if use_GPU
        y_ind = gpuArray(Y_H(1:p,ind:ind_end));
    else
        y_ind = Y_H(1:p,ind:ind_end);
    end
    a_hat_GS = GS(y_ind,X_H_computations,X_dag,GS_iters);
    residual_final=sqrt(sum(abs(y_ind-abs(X_H_computations*a_hat_GS)).^2,1))./sqrt(sum(abs(y_ind).^2,1));
    if use_GPU
        A(ind:ind_end,:) = gather(a_hat_GS');
        residual_vector(ind:ind_end) = gather(residual_final);
    else
        A(ind:ind_end,:) = a_hat_GS';
        residual_vector(ind:ind_end) = residual_final;
    end
    t1=toc(t0)
end
recon_time=toc(t00);
fprintf('Total Recon Time: %8.4f seconds\n', recon_time)
close(h);

A=single(gather(A));

switch k
    case 1
        save('./../Coherent_Data/AmpSLM_16x16/A_GS.mat','A','residual_vector','recon_time','-v7.3')
    case 2
        save('./../Coherent_Data/PhaseSLM_40x40/A_GS.mat','A','residual_vector','recon_time','-v7.3')
    case 3
        save('./../Coherent_Data/AmpSLM_64x64/A_GS.mat','A','residual_vector','recon_time','-v7.3')
end


