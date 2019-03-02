%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparing solution paths of linear regression under three different regularizations, 
% smoothly clipped absolute deviation (SCAD) penalty,
% minimax concave penalty (MCP), and L1 penalty (LASSO).
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
K0=2;                      % Number of nonzero components
rho0=K0/N;                 % Ratio of signal's nonzero components in synthetic data
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

% Set of amplitude nonconvexity parameter
Llam=100;  
lambda_max=ceil(max(abs(X'*Y)));
lambda_min=lambda_max*10^(-2);
rate=exp(log(lambda_min/lambda_max)/(Llam-1));
lambdaV=lambda_max*(rate.^[0:Llam-1]);

%% Soltuion path estimation with crosss-validation

% Solution path by lambda annealing with approximte CV by approximation 2
fitlasso=lassopath(Y,X,lambdaV);
fitscad=scadpath(Y,X,a,lambdaV);
fitmcp=mcppath(Y,X,a,lambdaV);

%%
lambda_c_scad=min(fitscad.lambda(fitscad.stab)); % Approximate CV instability point for SCAD
lambda_c_mcp=min(fitmcp.lambda(fitmcp.stab));    % Approximate CV instability point for MCP

% Cross validation error
figure;
hold on;
errorbar(fitlasso.lambda,fitlasso.cve(:,1),fitlasso.cve(:,2),'r+','MarkerSize',10);
errorbar(fitscad.lambda,fitscad.cve(:,1),fitscad.cve(:,2),'g*','MarkerSize',10);
errorbar(fitmcp.lambda,fitmcp.cve(:,1),fitmcp.cve(:,2),'bo','MarkerSize',10);
plot(lambda_c_scad*[1 1],[0,100],'g--','LineWidth',2.5);
plot(lambda_c_mcp*[1 1],[0,100],'b--','LineWidth',2.5);
set(gca,'XScale','Log');
ylim([0 1.*max(fitlasso.cve(:,1))]);
xlabel('\lambda');
ylabel('CV errors');
legend('LASSO(approx. 2)','SCAD','MCP','Instability(SCAD)','Instability(MCP)','Location','Best');

% Solution path
rng(4);
figure;
hold on;
for i=1:K0
    colorset=rand(1,3);
    plot(fitlasso.lambda,fitlasso.beta(i,:),'+:','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot(fitscad.lambda,fitscad.beta(i,:),'*--','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot(fitmcp.lambda,fitmcp.beta(i,:),'o-.','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot([lambda_min lambda_max],beta0(i)*[1 1],'-','Color',colorset,'LineWidth',2.5);
end
set(gca,'XScale','Log');
xlabel('\lambda');
ylabel('Solution path');
title('Solution path for finite signal component')
legend('LASSO','SCAD','MCP','True signal','Location','Best');

figure;
hold on;
for i=K0+1:K0+2
    colorset=rand(1,3);
    plot(fitlasso.lambda,fitlasso.beta(i,:),'+:','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot(fitscad.lambda,fitscad.beta(i,:),'*--','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot(fitmcp.lambda,fitmcp.beta(i,:),'o-.','Color',colorset,'LineWidth',2.5,'MarkerSize',8);
    plot([lambda_min lambda_max],beta0(i)*[1 1],'-','Color',colorset,'LineWidth',2.5);
end
set(gca,'XScale','Log');
xlabel('\lambda');
ylabel('Solution path');
ylim([-0.1 0.1]);
title('Solution path for no signal component')
legend('LASSO','SCAD','MCP','True signal','Location','Best');








