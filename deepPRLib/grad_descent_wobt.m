function [f,err_s,diff_s,z_set] = grad_descent_wobt(y_sq,A,z_0,x,eta,tol, p)
	z_set = [];
    [n,m] = size(A);
	z = z_0;
	Max_Iter =1e5;
	iter = 1;
	err = inf;
    fun_val = @(z) 1/2 * mean((y_sq - (abs(A'*z)).^2).^p ) ;
    
    %%

	while(err > tol) 
		 z_set(:,iter)  = z;%./norm(z); 
         yz             = A'*z;
		 z_old          = z;
		 f_old          = fun_val(z);
		 grad           = 1/m* A*( ( abs(yz).^2-y_sq ).^(p-1) .* yz ); 
         z              = z - eta*grad;
         
		 f = fun_val(z); 
		 iter_diff      = norm(z_old - z);
		 func_diff      = abs(f - f_old);
         diff_s(iter)   = iter_diff;
		 err            = norm(x - exp(-1i*angle(x'*z))*z)/norm(x);
         err_s(iter)    = err;
		 if iter_diff<=1e-10 || iter>Max_Iter
		     break;
		 end
% 		 fprintf('Iter = %d, Iter_Diff = %f, Func_Diff = %f, Err = %f \n',iter,iter_diff,func_diff,err);
		 
         iter = iter + 1;       
	end

end

