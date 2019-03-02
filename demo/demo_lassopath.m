%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the lambda annealing for obtaining a solution path 
% and of the approximate formula estimating cross-validation error
% for linear regression regularized by L1 penalty (LASSO)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Tomoyuki Obuchi
% Origial version was written on 2019 Mar. 2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%    See arXiv:1902.10375 and J. Stat. Mech. (2016) 053304
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

% Path to routine 
addpath('../routine/');

% Parameters for sample generation
N=200;                     % Model dimensionality (number of covariates)
alpha=0.5;                 % Ratio of dataset size to model dimensionaltiy
M=floor(alpha*N+10^(-12)); % Dataset size (number of response variables)
rho0=0.2;                  % Ratio of signal's nonzero components in synthetic data
K0=floor(rho0*N+10^(-12)); % Number of nonzero components
sigmaN2=0.1;               % Component-wise noise strength 
sigmaB2=1./rho0;           % Component-wise signal strength

% Sample generation
seed=1;
rng(seed);
beta0=zeros(N,1);
beta0(1:K0)=sqrt(sigmaB2)*randn(K0,1); % True signal
X0=randn(M,N);
X=X0;
for j=1:N
    av=mean(X0(:,j));
    nr=norm(X0(:,j)-av);
    X(:,j)=(X0(:,j)-av)/nr;            % Standardized design matrix 
end
Y=X*beta0+sqrt(sigmaN2)*randn(M,1);    
Y0=mean(Y);
Y=Y-Y0;                                % Centrizing response variable

%% Experiment

% Set of amplitude nonconvexity parameter
Llam=100;  
lambda_max=ceil(max(abs(X'*Y)));
lambda_min=lambda_max*10^(-2);
rate=exp(log(lambda_min/lambda_max)/(Llam-1));
lambdaV=lambda_max*(rate.^[0:Llam-1]);

%% Soltuion path estimation with crosss-validation

% Solution path by lambda annealing with approximte CV by approximation 2
fit=lassopath(Y,X,lambdaV);

% Solution path by lambda annealing with literal CV
kfold=M;
CVfit=lassopath(Y,X,lambdaV,'literal',kfold);

% Approximate cve by approximation 1
tic;
acvV=zeros(Llam,2);
for ilam=1:Llam
    [acve aerr]=acv_lasso(fit.beta(:,ilam),Y,X);
    acvV(ilam,:)=[acve aerr];
end
etime_acv=toc;

%% Plot   
disp(['elasped time for solution path estimation = ',num2str(fit.time(1)),' sec.']);
disp(['elasped time for approximate CV by approximation 1 = ',num2str(etime_acv),' sec.']);
disp(['elasped time for approximate CV by approximation 2 = ',num2str(fit.time(2)),' sec.']);
disp(['elasped time for literal ',num2str(kfold),'-fold CV = ',num2str(CVfit.time(2)),' sec.']);

figure;
hold on;
errorbar(fit.lambda,acvV(:,1),acvV(:,2),'ro','MarkerSize',10);
errorbar(fit.lambda,fit.cve(:,1),fit.cve(:,2),'g+','MarkerSize',10);
errorbar(CVfit.lambda,CVfit.cve(:,1),CVfit.cve(:,2),'b*','MarkerSize',10);
set(gca,'XScale','Log');
ylim([0 1.5*max(CVfit.cve(:,1))]);
xlabel('\lambda');
ylabel('CV errors');
legend('Approximation 1','Approximation 2','Literal','Location','Best');








