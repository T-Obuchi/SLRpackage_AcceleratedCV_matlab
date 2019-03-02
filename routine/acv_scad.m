function [acve aerr]=acv_scad(beta,Y,X,a,lambda)
%--------------------------------------------------------------------------
% acv_scad.m: An approximate formula of cross-validation error 
% for linear regression regularized by smoothly clipped absolute deviation (SCAD) penalty.
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute an approximate cross-validation (CV) error (cve) for  
%    estimated covariates coefficient beta at given nonconvexity parameters (a,lambda).
%    An approximate formula in [1] is applied.
%
% USAGE:
%    [acve aerr]=acv_scad(beta,Y,X,a,lambda);
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
%    For details of the formulation, see help of 'scadpath'
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
if a<1 
    error('the parameter "a" should be larger than unity');
end
beta_act=beta(S);
MASK=(abs(beta_act) < a*lambda).*(abs(beta_act) > lambda);
chi_tmp=inv( X_tmp'*X_tmp+diag(MASK)/(1-a) );

% Approximate CV error
for mu=1:M
    Factor=( 1 - X_tmp(mu,:)*chi_tmp*X_tmp(mu,:)' )^(-2);
    CV_all(mu)=Factor*RSS(mu);
end
acve=mean(CV_all);
aerr=std(CV_all)/sqrt(M); 
end
