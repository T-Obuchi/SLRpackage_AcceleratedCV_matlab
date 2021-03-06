function [beta,flag]=CCD_MCP_path(Y,X,a,lambdaV,theta,iter_max,beta_in);
% Preparation
[M,N]=size(X);
Llam=length(lambdaV);

% Default setting
if nargin < 7 || isempty(beta_in) 
    beta_in=zeros(N,1);
end
beta=zeros(N,Llam);
flag=zeros(1,Llam);


% Pathwise estimation
beta_tmp=beta_in;
for ilam=1:Llam
    lambda=lambdaV(ilam);
    output=CCD_MCP(Y,X',beta_tmp,lambda,a,theta,iter_max);
    beta_tmp=output(1:N)';
    beta(:,ilam)=beta_tmp;
    flag(ilam)=output(end);
    if flag(ilam)==1
        warning('CCD_MCP did not converge and break');
        break;
    end
end
        

end
