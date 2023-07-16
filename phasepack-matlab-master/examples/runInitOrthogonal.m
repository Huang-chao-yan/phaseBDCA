%%                   testinitOrthogonal.m

% This test file runs the Orthogonality Promoting initializer. The
% code builds a synthetic formulation of the Phase Retrieval problem, b =
% |Ax| and computes an estimate to x. The code finally outputs the
% correlation achieved.

% PAPER TITLE:
%              Phase retrieval with one or two diffraction patterns by
%              alternating projection with the null initializer.
% ARXIV LINK:
%              https://arxiv.org/pdf/1510.07379.pdf
% PAPER TITLE:
%              Solving Systems of Random Quadratic Equations via Truncated
%              Amplitude Flow.
% ARXIV LINK:
%              https://arxiv.org/pdf/1605.08285.pdf

% 1.) Each test script for an initializer starts out by defining the length
% of the unknown signal, n and the number of measurements, m. These
% mesurements can be complex by setting the isComplex flag to be true.

% 2.) We then build the test problem by generating random gaussian
% measurements and using b0 = abs(Ax) where A is the measurement matrix.

% 3.) We run x0 = initSpectral(A,[],b0,n,isTruncated,isScaled) which runs
% the initialier and and recovers the test vector with high correlation.



% PhasePack by Rohan Chandra, Ziyuan Zhong, Justin Hontz, Val McCulloch,
% Christoph Studer, & Tom Goldstein 
% Copyright (c) University of Maryland, 2017

%% -----------------------------START----------------------------------


clc
clear
close all

% Parameters
n = 500;           % number of unknowns
m = 8*n;          % number of measurements
isComplex = true;  % use complex matrices? or just stick to real?

%  Build the test problem
xt = randn(n,1)+isComplex*randn(n,1)*1i; % true solution
A = randn(m,n)+isComplex*randn(m,n)*1i;  % matrix
b0 = abs(A*xt);                          % data

% Invoke the truncated spectral initial method
x0 = initOrthogonal(A,[],b0,n);

% Calculate the correlation between the recovered signal and the true signal
correlation = abs(x0'*xt/norm(x0)/norm(xt));

fprintf('correlation: %f\n', correlation);