function [acve aerr]=saacv_lasso(beta,Y,X)
%--------------------------------------------------------------------------
% saacv_lasso.m: Further simplified approximate formula of cross-validation error 
% for linear regression with L1 penalty (LASSO)
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute an approximate cross-validation (CV) error (cve) for 
%    estimated covariates coefficient beta.
%    A simplified version of the approximate formula, 
%    approximation 2 in [2], is applied.
%
% USAGE:
%    [acve aerr]=saacv_lasso(beta,Y,X);
%
% INPUT ARGUMENTS:
%    beta        Estimated covariates' coefficients (N dimensional vector). 
%
%    Y           Response vector (M dimensional vector).
%
%    X           Matrix of covariates (M*N dimensional matrix).
%
% OUTPUT ARGUMENTS:
%    acve        Approximate CV error 
%
%    aerr        Approximate error bar for the CV error 
%
% DETAILS:
%    For details of the formulation, see help of 'lassopath'
%
% REFERENCES:
%    [1] Tomoyuki Obuchi and Ayaka Sakata: Cross validation in sparse linear 
%        regression with piecewise continuous nonconvex penalties and its acceleration
%        arXiv:1902.10375
%
%    [2] Tomoyuki Obuchi and Yoshiyuki Kabashima: Cross validation in LASSO 
%        and its acceleration
%        J. Stat. Mech. (2016) 053304
%
% DEVELOPMENT:
%    2 Mar. 2019: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M N]=size(X);
RSS=(Y-X*beta).^2/2;
S=abs(beta)>10^(-10);
rho=mean(S);
alpha=M/N;

% Approximate CV error
Factor=(alpha/(alpha-rho))^2;
CV_all=Factor*RSS;
acve=mean(CV_all);
aerr=std(CV_all)/sqrt(M); 
end
