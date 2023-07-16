% %  Chris Metzler.
% % Nov 17 2017
%% ImagingDemo.m reconstructsa sqrt(n)xsqrt(n) SLM pattern from a speckle pattern measured on the detector using the prVAMP algorithm.

%Add the gampMatlab toolbox to your path
addpath(genpath('~/gampmatlab'));

k=1;%Choose with dataset to use
clearvars -except k

%% Demonstrate Imaging
switch k
    case 1    
        load './../Coherent_Data/AmpSLM_16x16/A_prVAMP.mat'%Contains the transmission matrix and the residual vector
        load './../Coherent_Data/AmpSLM_16x16/YH_squared_test.mat'
        load './../Coherent_Data/AmpSLM_16x16/XH_test.mat'
        n=16^2;
    case 2
        load './../Coherent_Data/PhaseSLM_40x40/A_prVAMP.mat'%Contains the transmission matrix and the residual vector
        load './../Coherent_Data/PhaseSLM_40x40/YH_squared_test.mat'
        load './../Coherent_Data/PhaseSLM_40x40/XH_test.mat'
        n=40^2;
    case 3
        load './../Coherent_Data/AmpSLM_64x64/A_prVAMP.mat'%Contains the transmission matrix and the residual vector
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
[m,n]=size(A);
[U,S,V] = svd(A,'econ');
s = diag(S);
s = 1./s(:);
A_dag = (V.*s.')*U';
t0=tic;
GS_iters=200;
X_GS = GS(Y,A,A_dag,GS_iters);
t1=toc(t0)

%% Set prVAMP parameters
spars_init = 0.999; % initial sparsity rate (set near 1 !)
tuneDelayGamp = 0;%25;%25; % number of iterations to wait before tuning
tuneDelayVamp = 0;%25;%25; % number of iterations to wait before tuning
d=diag(S).^2;
vampOpt = VampGlmOpt;
vampOpt.nitMax = 2e2;%500;
% vampOpt.tol = 1e-4;%Worked fine for 16x16 and 40x40 data
vampOpt.tol = 1e-6;%1e-6;
vampOpt.damp = .8;%0.8; % try 0.8; 1 means no damping
vampOpt.dampGam = .5;%0.5; % try 0.5; 1 means no damping
vampOpt.dampConfig = [0,1,1,1,0,0, 0,0,0,0,0,1]; % best from dampTest
vampOpt.verbose = false;
vampOpt.silent = true;%Suppresses warnings. Necessary with use_GPU=true
vampOpt.gamMin=1e-8;%Default=1e-8
vampOpt.gamMax=1e8;%Default=1e11
vampOpt.U = U;
clear U;
vampOpt.V = V;
clear V;
vampOpt.d = d;
vampOpt.altUpdate = false;

rng(1);

NumOfRows=size(X_true,2);
nParallel=NumOfRows;
h=waitbar(0,'Computing Reconstruction...');
t00=tic;
for ind = 1:nParallel:NumOfRows
    ind
    vampOpt_thisiter=vampOpt;
    waitbar(ind/NumOfRows);
    t0=tic;
    
    ind_end=min(ind+nParallel-1,NumOfRows);
    L=length(ind:ind_end);%Should be nParallel except at the last iteration
    if use_GPU
        y_ind = gpuArray(Y(:,ind:ind_end));
    else
        y_ind = Y(:,ind:ind_end);
    end
    x_init=X_GS(:,ind:ind_end);%A was initialized with A_GS
    xvar_nz_init = var(x_init,1); % iniital nonzero-coef variance
    EstimIn = SparseScaEstim(...
           CAwgnEstimIn(zeros(1,L),xvar_nz_init,false,...
                        'autoTune',false,...
                        'mean0Tune',false,...
                        'tuneDim', 'col',...%If tuning is on, adapt the mean and variances for each of the L columns (rows of A) in the batch
                        'counter',tuneDelayGamp),...
           spars_init,false,'autoTune',false,'tuneDim','col','counter',tuneDelayGamp);
    wvar_init=ones(m,1)*mean(abs(y_ind-abs(A*x_init)).^2,1);%Per pixel noise estimate
    vampOpt_thisiter.r1init = x_init;
    vampOpt_thisiter.gam1xinit = 1./var(x_init,1);
    vampOpt_thisiter.p1init = A*x_init;
    vampOpt_thisiter.gam1zinit = 1./var(y_ind,1);
    this_recon_failed=0;
    try
        EstimOut = ncCAwgnEstimOut(y_ind,wvar_init,false,false);
        [x_hat,vampFin,optFin] = VampGlmEst2(EstimIn,EstimOut,A,vampOpt_thisiter);
    catch
        warning('Recon Failed Once');
        try
            EstimOut = ncCAwgnEstimOut(y_ind,2*wvar_init,false,false);
            [x_hat,vampFin,optFin] = VampGlmEst2(EstimIn,EstimOut,A,vampOpt_thisiter);
        catch
            warning('Recon Failed Twice');
            try
                EstimOut = ncCAwgnEstimOut(y_ind,4*wvar_init,false,false);
                [x_hat,vampFin,optFin] = VampGlmEst2(EstimIn,EstimOut,A,vampOpt_thisiter);
            catch
                x_hat=zeros(size(x_init));
                warning('Recon Failed Three Times. Using GS Solution.');
                this_recon_failed=1;
            end
        end
    end
    X_prVAMP(:,ind:ind_end)=gather(x_hat);
    t1=toc(t0);
    fprintf('%d images reconstructed in %8.4f seconds\n', L, t1)
end
recon_time=toc(t00);
fprintf('Total Recon Time: %8.4f seconds\n', recon_time)
close(h);

X_prVAMP=real(X_prVAMP);

figure;
for i=1:min(NumOfRows,5)
    subplot(2,5,i); imshow(reshape(X_prVAMP(:,i),sqrt(n),sqrt(n)),[]);
end

for i=1:min(size(X_true,2),5)
    subplot(2,5,i+5); imshow(reshape(X_true(:,i),sqrt(n),sqrt(n)),[]);
end
