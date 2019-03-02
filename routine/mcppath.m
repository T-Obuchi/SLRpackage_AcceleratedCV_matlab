function [fit]=mcppath(Y,X,a,lambdaV,cvoption,kfold,theta,iter_max,beta_in,seed);
%--------------------------------------------------------------------------
% mcppath.m: Estimating a solution path and cross-validation error 
% for linear regression with minimax concave penalty (MCP)
% by using lambda annealing
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Compute a solution path of linear regression 
%    regularized by minimax concave penalty (MCP).
%    The solution path is with respect to amplitude parameter lambda,
%    and the employed algorithm is the Cyclic Coordinate Discent (CCD).
%    As option, cross-validation (CV) error (cve) along the path can be computed.
%    Remark: An approximate formula of cve based on [1] is implemented, 
%    enabling a quick estimation of the leave-one-out cve.
%
% USAGE:
%    fit = mcppath(Y,X,a)
%    fit = mcppath(Y,X,a,lambdaV)
%    fit = mcppath(Y,X,a,lambdaV,cvoption,kfold,theta,iter_max,beta_in,seed)
%    (Use [] to apply the default value, e.g. 
%     fit = mcppath(Y,X,a,lambdaV,cvoption,[],theta,iter_max),
%     fit = mcppath(Y,X,a,lambdaV,[],[],[],[],beta_in),
%    )
%
% INPUT ARGUMENTS:
%    Y           Response vector (M dimensional vector).
%
%    X           Matrix of covariates (M*N dimensional matrix).
%
%    a           Switching parameter (a real number larger than unity).
%
%    lambdaV     Set of amplitude parameters (Llam dimensional vector).   
%                If not specified, a default set of Llam=100 is given.
%
%    cvoption    Option for CV. Three options are available:
%                'approximate': Approximate CV is conducted by the method of [1];
%                'literal': Literal CV is conducted;
%                'none': No CV is conducted.
%                The default setting is 'approximate'.
%
%    kfold       Fold number for CV. Only used when cvoption='literal'.
%                The default value is 10.
%
%    theta       Threshold to judge the convergence of the CCD algorithm.
%                The default value is 10^(-10).
%
%    iter_max    MAX iteration steps of the CCD algorithm. The default value is 10^5.
%
%    beta_in     Initial estimate of mean value of covariates' coefficients (N dimensional vector). 
%                Given as a zero vector if not specified. 
%
%    seed        Seed for random number generation (positive integer).
%                The default value is 1.
%
% OUTPUT ARGUMENTS:
%    fit         A structure.
%
%    fit.lambda  Set of amplitude parameters actually used 
%                in the pathwise estimation (Llam dimensional vector).
%
%    fit.beta    Estimated solution path (regression coefficients along the
%                set of lambda, N*Llam dimensional matrix).
%
%    fit.conv    Flags checking convergence of the CCD algorithm 
%                at each lambda (Llam dimensional vector). 
%                (0: converged, 1: not converged). 
% 
%    fit.tre     Training error for the solution path (Llam dimensional vector). 
%                The detailed definition including trivial coefficients is in DETAILS. 
%
%    fit.cve     CV error and its error bar 
%                for the solution path (Llam*2 dimensional matrix). 
%                fit.cve(:,1) gives the cve values and 
%                fit.cve(:,2) yields the error bars.
%                When cvoption='none', an empty array is returned.
%                The detailed definitionÅ@including trivial coefficients is in DETAILS. 
%
%    fit.stab    Flag for checking stability of the approximate CV 
%                (0: unstable, 1: stable). 
%                If the flag is zero, the corresponding approximate cve  
%                is not reliable and should be discarded. 
%                When cvoption='approximate', an Llam dimensional logical vector 
%                associated to all datapoints is returned, 
%                while when cvoption='literal' or 'none', 
%                an empty array is returned.
%                We recommend not to use the unstable region solutions.
%
%    fit.time    Elapsed time for estimating solution path (fit.time(1)) 
%                and for computing cve (fit.time(2)). 
%
% DETAILS:
%    The linear regression regularized by MCP is formulated as follows:
% 
%       \hat{beta}=argmin_{beta}
%           { (1/2)||Y-X*beta||_2^2 + \sum_{i}^{N}J(beta_i;lambda,a) }
%
%    where J denotes MCP whose functional form is
%
%       J(x;lambda,a)={ lambda*|x|-x^2/(2*a), ( |x|      <= a*lambda  )
%                     { a*lambda^2/2        , ( a*lambda < |x|        )
%
%    Each estimator at given (lambda,a) is computed by the CCD algorithm. 
%    We sweep the value of lambda from large to small ones, and at each
%    lambda the initial condition of beta for the CCD algorithm is given as 
%    the obtained estimator at the previous step with slightly larger lambda. 
%    This is termed lambda annealing in [1].
%    We define the training error for an estimator \hat{beta} as
%
%       tre=( 1/(2*M) )*||Y-X*\hat{beta}||_2^2.
% 
%    The CV error (cve) is also defined as follows: 
%    Divide given dataset (Y,X) into kfold sets of the (almost) same size, 
%    choose a test set (Y_j,X_{j.}) from the kfold sets, 
%    and the remaining sets constitute a training set (Y^{\j},X^{\j}).
%    The estimator for the training set (Y^{\j},X^{\j}) 
%    is denoted as \hat{beta}^{\j}. The cve for the test set is defined by
%
%       cve(j)=( 1/(2*D_j) )*||Y_j-X^{j.}*\hat{beta}^{\j} ||^2_2.
%
%    where D_j is the size of the test set (Y_j,X_{j.}). 
%    The mean of {cve(j)}_{j=1}^{kfold} gives the final estimate of cve,
%    and the standard deviation divided by sqrt(kfold) gives its error bar. 
%    The approximate CV gives an approximate estimate of these quantities
%    in the leave-one-out (kfold=M) case. 
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
%    9 Feb. 2019: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preparation
[M,N]=size(X);

% Parameters and default setting
if nargin < 3
    error('three input arguments needed at least');
end
if nargin < 4 || isempty(lambdaV) 
    % Default set of lambda
    Llam=100;  
    lambda_max=ceil(max(abs(X'*Y)));
    lambda_min=lambda_max*10^(-2);
    rate=exp(log(lambda_min/lambda_max)/(Llam-1));
    lambdaV=lambda_max*(rate.^[0:Llam-1]);
end
if nargin < 5 || isempty(cvoption)  
    cvoption='approximate';          % Approximate CV is a default setting
end
if nargin < 6 || isempty(kfold)  
    kfold=10;                        % Fold number for literal CV 
end
if nargin < 7 || isempty(theta) 
    theta = 1e-10;                   % convergence threshold 
end
if nargin < 8 || isempty(iter_max) 
    iter_max = 10^5;                 % MAX iteration number 
end
if nargin < 9 || isempty(beta_in) 
    beta_in=zeros(N,1);              % Initial regression coefficients
end
if nargin < 10 || isempty(seed) 
    seed=1;                          % Random seed
end


% Sort lambda values in the descending order
Llam=length(lambdaV);
[lambdaV_ord]=sort(lambdaV,'descend');


% cvoption check
if strcmp(cvoption,'approximate')
    disp('Approximate cross-validation conducted');
elseif  strcmp(cvoption,'literal')
    disp(['Literal ',num2str(kfold),'-fold cross-validation conducted']);
elseif  strcmp(cvoption,'none')
    disp('No cross-validation conducted');
else
    disp('Cvoption was not correctly specified');
    disp('Please choose one of the three below and retry:') 
    disp([' ''approximate'', ''literal'', ''none''']); 
end

% Evaluate solution path using lambda annealing
tic;
[beta,flag_conv]=CCD_MCP_path(Y,X,a,lambdaV_ord,theta,iter_max,beta_in);
etime_path=toc;


% Cross-validation
tic;
cve=zeros(Llam,2);
if strcmp(cvoption,'approximate')
    % Approximate CV
    for ilam=1:Llam
        [acve aerr]=acv_mcp(beta(:,ilam),Y,X,a,lambdaV_ord(ilam));
        cve(ilam,1:2)=[acve aerr];
    end
    
    % Stability check of the approximation
    [flag_unstable]=detect_instability(cve(:,1),cve(:,2));
    flag_stable=not(flag_unstable);
    
elseif  strcmp(cvoption,'literal') 
    % Literal CV
    rng(seed);
    S_all=randperm(M);
    D=floor(M/kfold+10^(-8));
    cve_all=zeros(Llam,kfold);
    for j=1:kfold
        % Dividing dataset
        S_test=sort(S_all(D*(j-1)+1:D*(j-1)+D));
        S_train=sort(setdiff(S_all,S_test));
        X_test=X(S_test,:);   % Design matrix for test set
        Y_test=Y(S_test);     % Response variables for test set
        X_train=X(S_train,:); % Design matrix for training set
        Y_train=Y(S_train);   % Response variables for training set

        % Solution path for training set
        [beta_tmp]=CCD_MCP_path(Y_train,X_train,a,lambdaV_ord,theta,iter_max,beta_in);

        % Test error for each test set
        for ilam=1:Llam
            cve_all(ilam,j)=sum((Y_test-X_test*beta_tmp(:,ilam)).^2)/(2*D);
        end
    end
    
    % CV error
    cve(:,1)=mean(cve_all,2);
    cve(:,2)=std(cve_all,1,2)/sqrt(kfold);
    flag_stable=[];
else    
    % No CV
    flag_stable=[];
    cve=[];
end
etime_cv=toc;


% Output
fit.lambda=lambdaV_ord;              % Set of amplitude parameter lambda
fit.beta  =beta;                     % Estimated regression coefficients
fit.conv  =flag_conv;                % Convergence flag of the CCD algorithm 
fit.tre   =sum((Y-X*beta).^2)/(2*M); % Training error
fit.cve   =cve;                      % CV error
fit.stab  =flag_stable;              % Stability flag of the approximate CV
fit.time  =[etime_path,etime_cv];    % Elapsed time for estimating solution path and CV error.
end
