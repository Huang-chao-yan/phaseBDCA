%Currently requries matlab 2017 (Code implicitly replicates vectors  that are addedto matrices).
%With older versions of Matlab nParallel must be 1.

% %  Chris Metzler.
% % Nov 17 2017
%% ConstructTM_prVAMP.m reconstructs an mxn transmission matrix using p distinct calibration/training x,y pairs using the prVAMP algorithm

%Add the gampMatlab toolbox to your path
addpath(genpath('~/gampmatlab'));

k=1;%Choose which transmission matrix to construct.
clearvars -except k

switch k
    case 1
        load ./../Coherent_Data/AmpSLM_16x16/XH_train.mat
        load ./../Coherent_Data/AmpSLM_16x16/YH_squared_train.mat
        load ./../Coherent_Data/AmpSLM_16x16/A_GS.mat
        A_GS=single(A);
        clear A
        n = 16^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=5e3;%Number of rows to compute in parallel. 
    case 2
        load ./../Coherent_Data/PhaseSLM_40x40/XH_train.mat
        load ./../Coherent_Data/PhaseSLM_40x40/YH_squared_train.mat
        load ./../Coherent_Data/PhaseSLM_40x40/A_GS.mat
        A_GS=single(A);
        clear A
        n = 40^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=1e3;%Number of rows to compute in parallel. 
    case 3
        load ./../Coherent_Data/AmpSLM_64x64/XH_train.mat
        load ./../Coherent_Data/AmpSLM_64x64/YH_squared_train.mat
        load ./../Coherent_Data/AmpSLM_64x64/A_GS.mat
        A_GS=single(A);
        clear A
        n = 64^2; % Size of SLM
        m = 256^2;% Size of Detector
        nParallel=1e3;%Number of rows to compute in parallel. 
end
NumOfRows = 256^2; % This controls how much of the TM is reconstructed 

%% Demonstrate Calibration (learning the transmission matrix)
fprintf('Made it to checkpoint 1\n');
Y_H = single(YH_squared_train).^.5;
clear YH_squared_train
X_H = single(XH_train)/255;
clear XH_train
A=A_GS(1:NumOfRows,:);
clear A_GS;
use_GPU=true;
if use_GPU
    X_H=gpuArray(X_H);
    A=gpuArray(A);
end
p_original=size(Y_H,1);
Resid_per_measurement = zeros(p_original,1);
for ind=1:nParallel:p_original
    ind_end=min(ind+nParallel-1,p_original);
    Resid_per_measurement(ind:ind_end)=gather(mean(abs(Y_H(ind:ind_end,:)-abs(X_H(ind:ind_end,:)*A')).^2,2));
end
Resid_per_measurement_sorted=sort(Resid_per_measurement);
%Throw out measurements with large residuals
bad_measurements = Resid_per_measurement>Resid_per_measurement_sorted(end-1*n);
Y_H = Y_H (~bad_measurements,:);
X_H = X_H(~bad_measurements,:);
p=size(X_H,1);% Number of measurements used to reconstruct TM

residual_final_vector=inf(NumOfRows,1);

fprintf('Made it to checkpoint 2\n');

%% Set prVAMP parameters

spars_init = 0.999; % initial sparsity rate (set near 1 !)
tuneDelayGamp = 0;%25; % number of iterations to wait before tuning
tuneDelayVamp = 0;%25; % number of iterations to wait before tuning
[U,S,V]=svd(X_H,'econ');
d=diag(S).^2;
clear S;
vampOpt = VampGlmOpt;
vampOpt.nitMax = 2e2;
vampOpt.tol = 1e-6;
vampOpt.damp = .8;%1 means no damping
vampOpt.dampGam = .5;%1 means no damping
vampOpt.dampConfig = [0,1,1,1,0,0, 0,0,0,0,0,1]; % best from dampTest
vampOpt.verbose = false;
vampOpt.silent = true;%Suppresses warnings.
vampOpt.gamMin=1e-8;%Default=1e-8
vampOpt.gamMax=1e8;%Default=1e11
vampOpt.U = U;
clear U;
vampOpt.V = V;
clear V;
vampOpt.d = d;
vampOpt.altUpdate = false;

fprintf('Made it to checkpoint 3\n');

residual_GS=residual_vector;
residual_vector=zeros(NumOfRows,1);
Bad_recons=0;
Good_recons=0;
failed_recons=0;
rng(1);

h=waitbar(0,'Computing transmission matrix...');
t00=tic;
for ind = 1:nParallel:NumOfRows
    ind
    waitbar(ind/NumOfRows);
    t0=tic;
    
    ind_end=min(ind+nParallel-1,NumOfRows);
    L=length(ind:ind_end);%Should be nParallel except at the last iteration
    if use_GPU
        y_ind = gpuArray(Y_H(1:p,ind:ind_end));
    else
        y_ind = Y_H(1:p,ind:ind_end);
    end
    a_init=A(ind:ind_end,:)';%A was initialized with A_GS
    wvar_hat=gpuArray(zeros(1,L));
    xvar_nz_init = var(a_init,1); % iniital nonzero-coef variance
    EstimIn = SparseScaEstim(...
           CAwgnEstimIn(zeros(1,L),xvar_nz_init,false,...
                        'autoTune',false,...
                        'mean0Tune',false,...
                        'tuneDim', 'col',...%If tuning is on, adapt the mean and variances for each of the L columns (rows of A) in the batch
                        'counter',tuneDelayGamp),...
           spars_init,false,'autoTune',false,'tuneDim','col','counter',tuneDelayGamp);
    wvar_init=ones(p,1)*mean(abs(y_ind-abs(X_H*a_init)).^2,1);%Per pixel noise estimate
    vampOpt.r1init = a_init;
    vampOpt.gam1xinit = 1./var(a_init,1);
    vampOpt.p1init = (X_H)*a_init;%y_ind;
    vampOpt.gam1zinit = 1./var(y_ind,1);
    a_hat=a_init;
    this_recon_failed=0;
    try
        EstimOut = ncCAwgnEstimOut(y_ind,wvar_init,false,false);
        [a_current,~,~] = VampGlmEst2(EstimIn,EstimOut,X_H,vampOpt);
    catch
        warning('Recon Failed Once');
        try
            EstimOut = ncCAwgnEstimOut(y_ind,2*wvar_init,false,false);
            [a_current,~,~] = VampGlmEst2(EstimIn,EstimOut,X_H,vampOpt);
        catch
            warning('Recon Failed Twice');
            try
                EstimOut = ncCAwgnEstimOut(y_ind,4*wvar_init,false,false);
                [a_current,~,~] = VampGlmEst2(EstimIn,EstimOut,X_H,vampOpt);
            catch
                a_current=zeros(size(a_init));
                warning('Recon Failed Three Times. Using GS Solution.');
                this_recon_failed=1;
            end
        end
    end
    init_residual=sqrt(sum(abs(y_ind-abs(X_H*a_init)).^2,1))./sqrt(sum(abs(y_ind).^2,1));
    this_residual=sqrt(sum(abs(y_ind-abs(X_H*a_current)).^2,1))./sqrt(sum(abs(y_ind).^2,1));
    good_rows=this_residual<init_residual;
    a_hat(:,good_rows)=a_current(:,good_rows);%Only replace the rows for which the residual is better than that of the initialization
    final_residual=sqrt(sum(abs(y_ind-abs(X_H*a_hat)).^2,1))./sqrt(sum(abs(y_ind).^2,1));
    Good_recons=Good_recons+sum(good_rows);
    Bad_recons=Bad_recons+(L-sum(good_rows));
    failed_recons=failed_recons+L*this_recon_failed;
    A(ind:ind_end,:)=gather(a_hat');
    residual_vector(ind:ind_end)=gather(final_residual(:));
    t1=toc(t0);
    fprintf('%d rows computed in %8.4f seconds\n', L, t1)
end
recon_time=toc(t00);
fprintf('Total Recon Time: %8.4f seconds\n', recon_time)
close(h);

A=single(gather(A));

switch k
    case 1
        save('./../Coherent_Data/AmpSLM_16x16/A_prVAMP.mat','A','residual_vector','recon_time','-v7.3')
    case 2
        save('./../Coherent_Data/PhaseSLM_40x40/A_prVAMP.mat','A','residual_vector','recon_time','-v7.3')
    case 3
        save('./../Coherent_Data/AmpSLM_64x64/A_prVAMP.mat','A','residual_vector','recon_time','-v7.3')
end


