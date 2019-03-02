%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demonstration of the lambda annealing for obtaining a solution path 
% and of the approximate formula estimating cross-validation error
% for linear regression regularized by smoothly clipped absolute deviation (SCAD) penalty.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% By Tomoyuki Obuchi
% Origial version was written on 2019 Feb. 8.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Method: 
%    See arXiv:1902.10375
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

% Switching parameter
a=5; 

% Set of amplitude parameter
Llam=100;  
lambda_max=ceil(max(abs(X'*Y)));
lambda_min=lambda_max*10^(-2);
rate=exp(log(lambda_min/lambda_max)/(Llam-1));
lambdaV=lambda_max*(rate.^[0:Llam-1]);

%% Soltuion path estimation with crosss-validation

% Solution path by lambda annealing with approximte CV
fit=scadpath(Y,X,a,lambdaV);

% Solution path by lambda annealing with literal CV
kfold=M;
CVfit=scadpath(Y,X,a,lambdaV,'literal',kfold);

%% Plot   
disp(['elasped time for solution path estimation = ',num2str(fit.time(1)),' sec.']);
disp(['elasped time for approximate CV  = ',num2str(fit.time(2)),' sec.']);
disp(['elasped time for literal ',num2str(kfold),'-fold CV = ',num2str(CVfit.time(2)),' sec.']);
lambda_c=min(fit.lambda(fit.stab)); % Instability point

figure;
hold on;
errorbar(fit.lambda,fit.cve(:,1),fit.cve(:,2),'ro','MarkerSize',10);
errorbar(CVfit.lambda,CVfit.cve(:,1),CVfit.cve(:,2),'b*','MarkerSize',10);
plot(lambda_c*[1 1],[0,100],'b--','LineWidth',2.5);
set(gca,'XScale','Log');
ylim([0 1.5*max(CVfit.cve(:,1))]);
xlabel('\lambda');
ylabel('CV errors');
legend('Approximate','Literal','Instability','Location','Best');








