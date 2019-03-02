function [acve aerr]=acv_lasso(beta,Y,X)
%--------------------------------------------------------------------------
% acv_lasso.m: An approximate formula of cross-validation error 
% for linear regression with L1 penalty (LASSO)
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute an approximate cross-validation (CV) error (cve) for 
%    estimated covariates coefficient beta.
%    An approximate formula, approximation 1 in [2], is applied.
%
% USAGE:
%    [acve aerr]=acv_lasso(beta,Y,X);
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

M=length(Y);
RSS=(Y-X*beta).^2/2;
S=abs(beta)>10^(-10);
CV_all=zeros(M,1);
X_tmp=X(:,S);

% Susceptibility estimation
chi_tmp=inv(X_tmp'*X_tmp);

% Approximate CV error
for mu=1:M
    Factor=( 1 - X_tmp(mu,:)*chi_tmp*X_tmp(mu,:)' )^(-2);
    CV_all(mu)=Factor*RSS(mu);
end
acve=mean(CV_all);
aerr=std(CV_all)/sqrt(M); 
end
