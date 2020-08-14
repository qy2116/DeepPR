clc;
close all;
clear all;
addpath('./deepPRLib/');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lp      = 2;                % lp norm for the objective function
ifSinit = 0;                % Spectral Init
if_real = 0;                % Real valued case or not
senseMat= 'Gauss';          %'Gauss', 'Uniform', 'CDP'
ifPlot  = 1;

n       = 10;               % data dimension
sr      = 2*(log(n))^3;      % sampling rate
m       = round(sr*n);      % number of sample point
depth   = 3;               % depth of over-parameteirzation

switch(senseMat)
    case 'Gauss'
        rA  = randn(n,m);
        iA  = randn(n,m);
        A   = 1/sqrt(2)*(rA+1i*iA);
    case 'Uniform'
        rA  = rand(n,m) - 0.5;
        iA  = rand(n,m) - 0.5;
        A   = 1/sqrt(2)*(rA+1i*iA);
    case 'CDP'
        A = [];
        A_temp   = dftmtx(n);
        for ind_mask = 1:sr
            Masks = randsrc(n,n,[1i -1i 1 -1]);
            A = [A, A_temp.*Masks];
        end
end
                        
if if_real ~= 0            
    % real sensing matrix
    A   = real(A);
    switch(senseMat)
        case 'Gauss'
            A = sqrt(2)*A;
        case 'Uniform'
            A = sqrt(2)*A;
    end
end
%% Optimal solution (up to global phase difference)
if if_real == 0
    x       =  (randn(n,1) + 1i*randn(n,1));
else
    x       = randn(n,1);
end
y_sq    = abs(A'*x).^2;     % generate the measurement

fun_val = @(p) 1/2 * mean((y_sq - (abs(A'*p)).^2).^lp ) ;

%% Initial Value
if if_real == 0
    z_0 = (randn(n,1) + 1i*randn(n,1))/4;
else
    z_0 = randn(n,1);
end

%% Spectral Init
if ifSinit == 1
    npower_iter = 50;                           % Number of power iterations
    z1 = z_0/norm(z_0,'fro');                   % Initial guess
    z0 = z1;
    for tt = 1:npower_iter                      % Power iterations
        z0 = A*(y_sq.* (A'*z0)); z0 = z0/norm(z0,'fro');
    end
    normest = sqrt(sum(y_sq)/numel(y_sq));      % Estimate norm to scale eigenvector
    z_1 = normest * z0;                         % Apply scaling
else
    z_1 = z_0;
end
%%
tol     = 1e-15;        %stopping criteria for gradient-descent algorithm

etaV    = 1e-4;
    
%% Overparameterized formulation gradient descent
%% MOP, Matrix Over-Parameterization Depth 1
tic
[~, err_acc_mop, ~ ,~, z_gd_set1] = grad_descent_acc_deepsqmat_wobt(y_sq, A, z_1, x, etaV, tol, depth, lp);
t_mop = toc;
    
for ind = 1:size(z_gd_set1,2)
    f_acc_mop(ind) = fun_val(z_gd_set1(:,ind));
end  
    
    
%% SOP, Scalar Over-Parameterization
tic
[~, err_acc_sop,~,~,~,z_gd_set2] = grad_descent_acc_deepscalar_wobt(y_sq, A, z_1, x, etaV, tol, 1, lp);
t_sop = toc;
for ind = 1:size(z_gd_set2,2)
    f_acc_sop(ind) = fun_val(z_gd_set2(:,ind));
end
    
%% vanilla gradient descent
tic
[~, err_gd,~, z_gd_set4] = grad_descent_wobt(y_sq, A, z_1, x, etaV, tol, lp);
t_gd = toc;

for ind = 1:size(z_gd_set4,2)
    f_gd(ind) = fun_val(z_gd_set4(:,ind));
end

%% Draw Figures
figure(1);
semilogy(f_gd,'Color',[0, 0, 0, 1],'LineWidth',4.5);
hold on
semilogy(f_acc_sop,'Color',[0, 0, 1, 1],'LineWidth',4.5);
semilogy(f_acc_mop,'Color',[1, 0, 0, 1],'LineWidth',4.5);
grid on
xlabel('Iteration Number','FontSize',15);
ylabel('Recovery Error','FontSize',15);
legend('DeepPR-0',['DeepPR-SOP-',num2str(depth)],['DeepPR-MOP-',num2str(depth)],'Location','northeast');%
legend boxoff
%%
fprintf('t_gd = %d, t_sop = %f, t_mop = %f \n',t_gd,t_sop,t_mop);
