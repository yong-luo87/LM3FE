function [matW_new, vecB_new, obj_final] = optimizeW(matXs, matY, X_inf, matUs, theta, matW, vecB, set, para)
% -------------------------------------------------------------------------
% Optimization of the classification matrix W
% -------------------------------------------------------------------------

maxit = 100;
sigma = para.sigma;
epsilon = 1e-3;

% -------------------------------------------------------------------------
% Compute the transformed feature matrix and correpsonding kernel matrix
% -------------------------------------------------------------------------
matZ = zeros(set.nbP, set.nbL);
for v = 1:set.nbV
    matZ = matZ + theta(v) * matUs{v}' * matXs{v}';
end
matZe = [matZ; ones(1, set.nbL)]; clear matZ

% matZ1 = zeros(set.nbC, set.nbL);
% for v = 1:set.nbV
%     for n = 1:set.nbL
%         matZ1(:, n) =  matZ1(:, n) + theta(v) * matUs{v}' * singleTrainFeaL{v}(n, :)';
%     end
% end

% -------------------------------------------------------------------------
% Train NeSVM (smoothed primal SVM) for each concept
% -------------------------------------------------------------------------

tic; fprintf('Optimizing W{p} and bias b{p} ... ');

% -----------------------------------------------------------
% Calculate the lipschitz constant
% -----------------------------------------------------------
lipsc = computeLipscWp(matZe, X_inf, sigma, set, para);

% -----------------------------------------------------------
% Nesterov's optimal gradient method for SVM
% -----------------------------------------------------------
for p = 1:set.nbP
    fprintf('%d ', p);
    vecWpBp = [zeros(size(matW(:,p))); 0]; % vecWp = matW(:,p);
    vecWpBp_guess = [zeros(size(matW(:,p))); 0]; % vecWp_guess = matW(:,p);
    
    % -------------------------------------------------------
    % Initialization of the objective, gradient
    % -------------------------------------------------------
    [obj(1,1), grad(:,1), obj_Phi] = computeObjGradWp(matZe, matY(:,p), X_inf, vecWpBp, sigma, set, para);
    
    loop = 1; t = 1;
    while loop
        % ---------------------------------------------------
        % Solving the two auxiliary optimization problems
        % ---------------------------------------------------
        y = vecWpBp - (1.0/lipsc)*grad(:,t);
        tempGrad = zeros(size(grad,1), 1);
        for i = 1:t
            tempGrad = tempGrad + (i/2.0)*grad(:,i);
        end
        z = vecWpBp_guess - (1.0/lipsc)*tempGrad;
        clear tempGrad
        
        % ------------------------------------------------------
        % Project the result to [0, 1]
        % ------------------------------------------------------
        % z = (z - min(z)) / (max(z) - min(z));
        
        % ------------------------------------------------------
        % Update the solution, objective and gradient
        % ------------------------------------------------------
        vecWpBp_new = (2.0/(t+3))*z + ((t+1)*1.0)/(t+3)*y; clear y z
        
        t = t + 1;
        % sigma = para.sigma / t;
        %     if sigma > 1
        %         sigma = para.sigma / t;
        %     end
        
        [obj(t,1), grad(:,t), obj_Phi] = computeObjGradWp(matZe, matY(:,p), X_inf, vecWpBp_new, sigma, set, para);
        
        % ------------------------------------------------------
        % Check convergence
        % ------------------------------------------------------
        obj_diff = abs(obj(t,1) - obj(t-1,1)) / abs(obj(t,1) - obj(1,1));
        if abs(obj(t,1) - obj(1,1)) < eps || obj_diff <= epsilon || t >= maxit
            loop = 0;
        end
        
        % ------------------------------------------------------
        % Update variables
        % ------------------------------------------------------
        if loop
            clear vecWpBp
            vecWpBp = vecWpBp_new;
            clear vecWpBp_new
        end
    end
    
    % Store the optimized Wp and Bp
    matW(:, p) = vecWpBp_new(1:set.nbP); matW_new(:, p) = vecWpBp_new(1:set.nbP);
    vecB(p) = vecWpBp_new(set.nbP+1); vecB_new(p) = vecWpBp_new(set.nbP+1);
    obj_new(p) = obj(t,1);
    clear vecWpBp_new obj grad
end
obj_final = sum(obj_new(:));
% obj_final = (1.0/set.nbP) * sum(obj_new(:));
timecost = toc;
fprintf('Finished! timecost = %.4f s \n', timecost);

end



function [obj, grad, obj_Phi] = computeObjGradWp(matZe, vecY, X_inf, vecWpBp, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the objective value and gradient
% -------------------------------------------------------------------------

diagY = diag(vecY);
numer = diagY * (matZe' * vecWpBp);
denom = sigma * X_inf;

idx1 = find(numer > 1);
idx2 = find(numer < (1-denom));
idx3 = setdiff((1:set.nbL)', [idx1; idx2]);

nu = zeros(set.nbL, 1);
nu(idx2) = 1;
nu(idx3) = (1 - numer(idx3)) ./ denom(idx3);

obj_Phi_temp = zeros(set.nbL, 1);
obj_Phi_temp(idx2) = (1 - numer(idx2)) - 0.5*denom(idx2);
obj_Phi_temp(idx3) = (1 - numer(idx3)).^2 ./ (2.0*denom(idx3));
obj_Phi = sum(obj_Phi_temp(:)); clear obj_Phi_temp
% obj_Phi = (1.0/set.nbL) * sum(obj_Phi_temp(:)); clear obj_Phi_temp
clear numer denom idx1 idx2 idx3

grad_Phi = - matZe * diagY * nu;
grad_Phi = grad_Phi;
% grad_Phi = (1.0/set.nbL) * grad_Phi;
clear diagY nu

obj = obj_Phi + para.gammaA * (vecWpBp(1:set.nbP)'*vecWpBp(1:set.nbP));
% obj = obj_Phi + para.gammaA * set.nbP * (vecWpBp(1:set.nbP)'*vecWpBp(1:set.nbP));

grad_Omega = 2*para.gammaA*vecWpBp; grad_Omega(set.nbP+1) = 0;
% grad_Omega = 2*para.gammaA*set.nbP*vecWpBp; grad_Omega(set.nbP+1) = 0;

grad = grad_Phi + grad_Omega;

end

function lipsc = computeLipscWp(matZe, X_inf, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the lipschitz constant
% -------------------------------------------------------------------------

tempVec = zeros(set.nbL, 1);
for n = 1:set.nbL
    tempVec(n) = norm(matZe(1:set.nbP,n)*matZe(1:set.nbP,n)', 2);
    tempVec(n) = tempVec(n) / X_inf(n);
end
lipsc = (set.nbL*1.0) / sigma * max(tempVec(:)) + 2*para.gammaA;
% lipsc = 1.0 / sigma * max(tempVec(:)) + 2*para.gammaA*set.nbP;

end

