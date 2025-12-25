function [x, numit] = apgd(gradf, prox_psi, gamma, L, x0, tol, maxit) 
   % APGD using fixed step size of alpha = 1/L

    x_prev = x0;
    x_curr = x0;
    y_curr = x0;
    
    rho_prev = 0;
    alpha = 1 / L;

    for k = 1:maxit
        grad = L * (y_curr - prox_psi(y_curr - alpha * gradf(y_curr), alpha));
        
        % Check convergence
        if norm(grad) < tol
            numit = k;
            x = y_curr; 
            return;
        end
        
        x_curr = y_curr - alpha * grad;
        
        % use quadratic formula
        b = 1 - rho_prev^2;
        rho_curr = (-b + sqrt(b^2 + 4)) / 2;
        
        beta = rho_curr * rho_prev^2;
        
        y_next = x_curr + beta * (x_curr - x_prev);
        
        % update
        x_prev = x_curr;
        y_curr = y_next;
        rho_prev = rho_curr;
    end

    % If max iterations reached without convergence
    numit = maxit;
    x = x_curr;
end