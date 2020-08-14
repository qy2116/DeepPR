function [z,f,err_s, w_s ,diff_s, z_set] = grad_descent_acc_deepscalar_wobt(y_sq,A,z_0,x,eta,tol,sz,p)
z_set       = [];
[n,m]       = size(A);

w           = ones(1, sz);

z           = z_0;
z           = z/prod(w);

Max_Iter    = 1e5;
w_s         = zeros(Max_Iter,sz);

iter        = 1;
err         = inf;

fn_val = @(z) 1/2 * mean((abs(A'*z).^2 - y_sq).^p);

while(err > tol)
    z_set(:,2*iter-1) = prod(w)*z;
    yz          = A'* prod(w)*z;
    z_old       = prod(w)* z;
    f_old       = fn_val(prod(w)*z);
    gradz       = 1/m*prod(w)*A*((abs(yz).^2-y_sq ).^(p-1).*yz);
    
    for i=1:sz
        grad{i} = 1/m*((abs(yz))'.^2/w(i)*(abs(yz).^2-y_sq ).^(p-1));
    end

    z = z -  eta * gradz;
    z_set(:,2*iter) = prod(w)*z;
    err = norm(x -  exp(-1i*angle(x'* z_set(:,2*iter)))* (z_set(:,2*iter)))/norm(x);
    err_s(2*iter) = err;
    
    alpha = eta*ones(1,sz);
    for i=1:sz
        w(i) = w(i) -  alpha(i)*grad{i};
    end
    
    w_s(iter,:) = w;
    f = fn_val(prod(w)*z);
    iter_diff = norm(z_old - prod(w)*z);
    diff_s(iter) = iter_diff;
    err = norm(x -  exp(-1i*angle(x'* prod(w) * z))* (prod(w) *z))/norm(x);
    if norm( prod(w) * z-z_old)<= 1e-10 || iter>Max_Iter
        break;
    end
%     fprintf('Iter = %d, Iter_Diff = %f, Func_Diff = %f, Err = %f \n',iter,iter_diff,func_diff,err);
    iter = iter + 1;
end

w_s = w_s(1:iter,:);
end

