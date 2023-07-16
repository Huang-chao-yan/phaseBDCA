clear all;
clear memery
% close all;
%by chaoyan, Aug 2021
%add PR_ADMM, phasepack-matlab-master into the path
%% Set Parameters
randn('state',2021);
rand ('state',2021);
% Number of masks
L  = 2;
ParamsTV.L=L;
%%  Read Image
% % Below X is n1 x n2 x 3; i.e. we have three n1 x n2 images, one for each of the 3 color channels
% namestr = 'cameraman' ;stanstr = 'tif'      ;
  namestr = '01' ;stanstr = 'tif'      ;
% namestr = 'Noiseless_1maskDDWF_acetaminophen' ;stanstr = 'jpg'      ;
% namestr = 'pirate' ;stanstr = 'jpg'      ;
% namestr = 'woman_darkhair' ;stanstr = 'jpg'      ;
%namestr = 'lena' ;stanstr = 'png'      ;
X = mat2gray(imread([namestr,'.',stanstr]));
X=rgb2gray(X);
sigma=10;ParamsTV.gamma=5e3;ParamsTV.lambda=2e3;ParamsTV.elp=1e-4;
lambdaWF=1200;%1150;
% sigma is  the noise level.
ParamsTV.flag=1;
ParamsTV.const=1;
ParamsTV.maxDCA = 20;%20
ParamsTV.nIter = 15;%100
ParamsTV.tol=1e-10;
ParamsTV.oit=1;
ParamsTV.tau=1;
ParamsTV.HIO_iter=50;
n1      = size(X,1)                               ;
n2      = size(X,2)                                ;

%% Make masks and linear sampling operators
Masks = zeros(n1,n2,L);  % Storage for L masks, each of dim n1 x n2
for ll = 1:L, Masks(:,:,ll) = randsrc(n1,n2,[1i -1i 1 -1]); end
temp = rand(size(Masks));
Masks = Masks .* ( (temp <= 0.2)*sqrt(3) + (temp > 0.2)/sqrt(2) );       
% Make linear operators;
A = @(I) fft2(Masks.*reshape(repmat(I,[1 L]),size(I,1),size(I,2),L)); % Input is n1 x n2 image, output is n1 x n2 x L array
At= @(Y) sum(conj(Masks).*ifft2(Y), 3)*size(Y,1)*size(Y,2);
AtA = @(I) At(A(I));

AtAER=sum(Masks .*conj(Masks), 3)*n1*n2; %%n1*n2 is the scale  of FFT and IFFT


%% Generate the obersavations 
x = squeeze(X(:,:,1)); % Image x is n1 x n2
W = A(x);        % Measured data
YGauss=abs(W) +sigma*randn(n1,n2,L);
%IPhase0=log(abs(b0) + eps);
Params.n1=n1;Params.n2=n2;

% Projection operator
PM=@(Z) YGauss.*sign(Z);
PS=@(Z) A(real(At(Z)./AtAER));
disp('initial ER')
tic
u0=ER_image(n1,n2,A,At,AtAER,PM,PS,40);
toc
snr_ini=snrComptC(u0,x);


disp('starting the bdca WF model')
tic;
% [uWF_bdca,inerWF_bdca,ERWF_bdca] = solverWF_bdca(YGauss,x, A, At,ParamsTV,AtA,u0);
% [uWF_bdca,inerWF_bdca,ERWF_bdca] = solverWF(YGauss,x, A, At,ParamsTV,AtA,u0);
[uWF_bdca,inerWF_bdca,ERWF_bdca] = solverWFkappa(YGauss,x, A, At,ParamsTV,AtA,u0,lambdaWF);
toc;

disp('starting the bdca TV model')
tic;
[uPR_bdca,inerER_bdca,ER_bdca] = solverTV_bdca(YGauss,x, A, At,ParamsTV,AtA,u0);
toc;
%%%%%%%%%%%%%%%%%%raar
tol=1e-14; % torlance
delta=0.75;
iter_max=50;
disp('starting the RAAR model')
tic;
[u_RAAR,error_RAAR,snr_RAAR]=solveRAAR(A,At,AtAER,PS,PM,x,YGauss,iter_max,tol,u0,delta);
toc;



opts = struct;                % Create an empty struct to store options
opts.algorithm = 'rwf';       % Use the truncated Wirtinger flow method to solve the retrieval problem.  Try changing this to 'Fienup'.
opts.initMethod = 'optimal';  % Use a spectral method with optimized data pre-processing to generate an initial starting point for the solver  
opts.tol = 1e-3;              % The tolerance - make this smaller for more accurate solutions, or larger for faster runtimes
opts.verbose = 2;             % Print out lots of information as the solver runs (set this to 1 or 0 for less output)
opts = manageOptions(opts);
opts.recordMeasurementErrors=0;
opts.recordReconErrors=0;
opts.recordResiduals=1;
opts.alpha_lb=0.1;opts.reweightPeriod=20;opts.betaChoice='HS';opts.searchMethod='steepestDescent';
opts.alpha_ub=5;opts.alpha_h=6;opts.truncationPeriod=20;

%%%%%%%%%%%%%%%raf
disp('starting the RAF model')
tic
[u_RAF, outs_raf] = solveRAF(A, At, YGauss, u0, opts);
toc
SNR_raf=snrCompt(u_RAF,x);

%%%%%%%%%%%%%%%%rwf
disp('starting the RWF model')
opts.eta=0.9;
tic
[u_RWF, outs_rwf] = solveRWF(A,At,YGauss,u0,opts);
toc
SNR_rwf=snrCompt(u_RWF,x);

%%%%%%%%%%%%%%%%taf
disp('starting the TAF model')
opts.gamma=0.7;
tic
[u_TAF, outs_taf] = solveTAF(A, At,YGauss, u0, opts);
toc
SNR_taf=snrCompt(u_TAF,x);
% disp('starting the dca model')
% tic;
% [uPR_dca] = solverTV_dca(YGauss,x, A, At,ParamsTV,AtA);
% toc;

%%%%%%%%%%cda
opts.indexChoice='random';
disp('starting the CDA model')
tic
[u_CDA, outs] = solveCoordinateDescent(A, At, YGauss, u0, opts);
toc
SNR_cda=snrCompt(u_CDA,x);


%%%%%%%%%%twf
disp('starting the TWF model')
tic
[u_TWF, outs_twf] = solveTWF(A, At, YGauss, u0, opts);
toc
SNR_twf=snrCompt(u_TWF,x);


%%%%%%solveWF
disp('starting the WF model')
opts.regularizationPara=0.1;
tic
[u_WF, outs_wf] =solveWirtFlow(A, At, YGauss, u0, opts);
toc
SNR_wf=snrCompt(u_WF,x);

%%%%%%%%HIO
tic
[u_HIO,error_HIO,snr_HIO]=solveHIO(A,At,AtAER,PS,PM,x,YGauss,iter_max,tol,u0,delta);
toc
SNR_hio=snrCompt(u_HIO,x);

%%%%%%%%%%%%%%%%ddwf
% opts.initial=u0; operator.A=A;
% operator.AT=At;
% operator.kappa=2.2;
% operator.N1=n1;
% operator.N2=n2;opts.truth=x;
% tic
 % [x,out] = DDWF(YGauss,operator,opts);
% %[xRetrieved, outs] = solveDDWF(data,operator,method,param);
% %[u_DDWF, outs_ddwf] = solveDDWF(A, At,YGauss, u0, opts);
% toc
% SNR_ddwf=snrCompt(u_DDWF,x);


%%
% SNRPR_dca=snrCompt(uPR_dca,x);
% ssim(uPR_dca,x);
SNRPR_bdca=snrCompt(uPR_bdca,x);
ssim(uPR_bdca,x);
SNRWF_bdca=snrCompt(uWF_bdca,x);
ssim(uWF_bdca,x);
% figure;imshow([x;uPR_dca;uPR_bdca])

figure;
subplot(331);imshow(x);title(['original']);
subplot(332);imshow(u0);title(['ini psnr=',num2str(snr_ini),'dB']);
subplot(334);imshow(uPR_bdca);title(['TV psnr=',num2str(SNRPR_bdca),'dB']);
subplot(333);imshow(uWF_bdca);title(['WF psnr=',num2str(SNRWF_bdca),'dB']);
subplot(335);imshow(u_RAAR);title(['RAAR psnr=',num2str(snr_RAAR),'dB']);
subplot(336);imshow(real(u_RAF));title(['RAF psnr=',num2str(SNR_raf),'dB']);
subplot(337);imshow(real(u_RWF));title(['RWF psnr=',num2str(SNR_rwf),'dB']);
subplot(338);imshow(real(u_TAF));title(['TAF psnr=',num2str(SNR_taf),'dB']);
subplot(339);imshow(real(u_CDA));title(['CDA psnr=',num2str(SNR_cda),'dB']);

imwrite(u0,['result_J2_sig10/',namestr,'ER.png'])
imwrite(uWF_bdca,['result_J2_sig10/',namestr,'TF.png'])
imwrite(uPR_bdca,['result_J2_sig10/',namestr,'TV.png'])
imwrite(u_RAAR,['result_J2_sig10/',namestr,'RAAR.png'])
imwrite(real(u_RAF),['result_J2_sig10/',namestr,'RAF.png'])
imwrite(real(u_RWF),['result_J2_sig10/',namestr,'RWF.png'])
imwrite(real(u_TAF),['result_J2_sig10/',namestr,'TAF.png'])
imwrite(real(u_CDA),['result_J2_sig10/',namestr,'CDA.png'])