function [f,err_s, w_s ,diff_s, z_set] = grad_descent_acc_deepsqmat_wobt(y_sq,A,z_0,x,eta,tol, numMat, p)
z_set       = [];
[n,m]       = size(A);

for i=1:numMat
    W(:,:,i) =  eye(n) ;
end

z           = z_0;

%%
Max_Iter    = 1e5;
w_s         = cell(Max_Iter,1);

iter        = 1;
err         = inf;

fun_val = @(z,w) 1/2 * mean((abs(A'*prodMat(w)*z).^2 - y_sq).^p);

while(err > tol)
    We          = prodMat(W);
    
    z_set(:,2*iter-1) = We*z;
    err = norm(x -  exp(-1i*angle(x'* z_set(:,2*iter-1)))* (z_set(:,2*iter-1)))/norm(x);
    err_s(2*iter-1) = err;
    
    yz          = A'*We*z;
    z_old       = z_set(:,end);
    f_old       = fun_val(z,W);
    gradz       = 1/m*We'*A*((abs(yz).^2-y_sq ).^(p-1).*yz);
    
    
    seq = 1:numMat;
    grad = zeros(n,n,numMat);
    for ind=1:numMat
        i = seq(ind);
        yz          = A'*We*z;
        if i-1 < 1
            We1 = 1;
        else
            We1 = prodMat(W(:,:,1:i-1));
        end
        
        if numMat < i + 1
            We2 = 1;
        else
            We2 = prodMat(W(:,:,i+1:numMat));
        end
        
        grad(:,:,i) = 1/m * We2' * A * ( ( abs(yz).^2-y_sq ).^(p-1) .* yz ) * (We1*z)';       
        
    end
    
    z = z - eta*gradz;
    
    z_set(:,2*iter) = We*z;
    err = norm(x -  exp(-1i*angle(x'* z_set(:,2*iter)))* (z_set(:,2*iter)))/norm(x);
    err_s(2*iter) = err;
    
    W = W - eta*grad;
    
    w_s{iter} = W;
    f = fun_val(z,W);
    iter_diff = norm(z_old - z_set(:,end));
    diff_s(iter) = iter_diff;
    func_diff = abs(f - f_old);
    if iter_diff<=1e-10 || iter>Max_Iter
        break;
    end
%     fprintf('Iter = %d, Iter_Diff = %f, Func_Diff = %f, Err = %f \n',iter,iter_diff,func_diff,err);
    
    iter = iter + 1;
end
w_s = w_s(1:iter,:);
end

