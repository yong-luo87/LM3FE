function theta_new = optimizeTheta(matXs, matY, X_inf, matW, vecB, matUs, theta, set, para)
% -------------------------------------------------------------------------
% Optimization of the view combination coefficients theta
% -------------------------------------------------------------------------

maxit = 100;
sigma = para.sigma;
epsilon = 1e-3;

% theta = zeros(size(theta));
% theta_guess = zeros(size(theta));
theta = theta;
theta_guess = theta;

matZVs = cell(set.nbV, 1);
for v = 1:set.nbV
    matXs{v} = matXs{v}';
    matZVs{v} = matXs{v}' * matUs{v} * matW;
end
matZPs = cell(set.nbP, 1);
for p = 1:set.nbP
    tempZp = [];
    for v = 1:set.nbV
        tempZp = [tempZp matZVs{v}(:,p)];
    end
    matZPs{p} = tempZp'; clear tempZp
end
clear matZVs

% -------------------------------------------------------------------------
% Optimize theta using the Nesterov's gradient method
% -------------------------------------------------------------------------

tic; fprintf('Optimizing theta ... ');
% ----------------------------------------------------------
% Calculate the lipschitz constant
% ----------------------------------------------------------
lipsc = computeLipscTheta(matZPs, X_inf, sigma, set, para);

% ----------------------------------------------------------
% Initialization of the objective
% ----------------------------------------------------------
[obj(1,1), obj_Phi] = computeObjTheta(matZPs, matY, X_inf, vecB, theta, sigma, set, para);
grad(:,1) = computeGradTheta(matZPs, matY, X_inf, vecB, theta, sigma, set, para);

loop = 1; t = 1;
while loop
    % ------------------------------------------------------
    % Solving the two auxiliary optimization problems
    % ------------------------------------------------------
    y = theta - (1.0/lipsc)*grad(:,t);
    tempGrad = zeros(size(grad,1), 1);
    for i = 1:t
        tempGrad = tempGrad + (i/2.0)*grad(:,i);
    end
    z = theta_guess - (1.0/lipsc)*tempGrad;
    clear tempGrad
    
    % ------------------------------------------------------
    % Project the result to [0, 1]
    % ------------------------------------------------------
    % z = (z - min(z)) / (max(z) - min(z));
    
    % ------------------------------------------------------
    % Update the solution
    % ------------------------------------------------------
    theta_new = (2.0/(t+3))*z + ((t+1)*1.0)/(t+3)*y; clear y z
    
    t = t + 1;
    % sigma = para.sigma / t;
    %     if sigma > 1
    %         sigma = para.sigma / t;
    %     end
    
    % ------------------------------------------------------
    % Update the objective value and gradient
    % ------------------------------------------------------
    [obj(t,1), obj_Phi] = computeObjTheta(matZPs, matY, X_inf, vecB, theta_new, sigma, set, para);
    grad(:,t) = computeGradTheta(matZPs, matY, X_inf, vecB, theta_new, sigma, set, para);
    
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
        clear theta
        theta = theta_new;
        clear theta_new
    end
end
timecost = toc;
fprintf('Finished! timecost = %.4f s \n', timecost);

end



function [obj, obj_Phi] = computeObjTheta(matZPs, matY, X_inf, vecB, theta, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the objective value
% -------------------------------------------------------------------------

denom = sigma*X_inf;
obj_Phi_temp = zeros(set.nbL, set.nbP);

for p = 1:set.nbP
    vecYp = matY(:, p); diagYp = diag(vecYp);
    numer_p = matZPs{p}'*theta + vecB(p); numer_p = diagYp * numer_p;
    
    idx1 = find(numer_p > 1);
    idx2 = find(numer_p < 1-denom);
    idx3 = setdiff((1:set.nbL)', [idx1; idx2]);
    
    obj_Phi_temp(idx2, p) = (1-numer_p(idx2)) - 0.5*denom(idx2);
    obj_Phi_temp(idx3, p) = (1-numer_p(idx3)).^2 ./ (2.0*denom(idx3));
    clear numer_p denom_p idx1 idx2 idx3
    clear vecYp diagYp
end

obj_Phi = sum(obj_Phi_temp(:)); clear obj_Phi_temp
% obj_Phi = (1.0/(set.nbL*set.nbP)) * sum(obj_Phi_temp(:)); clear obj_Phi_temp
obj = obj_Phi + para.gammaC*(theta'*theta);

end

function grad = computeGradTheta(matZPs, matY, X_inf, vecB, theta, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the gradient
% -------------------------------------------------------------------------

denom = sigma*X_inf;
grad_Phi = zeros(size(theta));

for p = 1:set.nbP
    vecYp = matY(:, p); diagYp = diag(vecYp);
    numer_p = matZPs{p}'*theta + vecB(p); numer_p = diagYp * numer_p;
    
    idx1 = find(numer_p > 1);
    idx2 = find(numer_p < 1-denom);
    idx3 = setdiff((1:set.nbL)', [idx1; idx2]);
    
    nu_p = zeros(set.nbL, 1);
    nu_p(idx2) = 1;
    nu_p(idx3) = (1 - numer_p(idx3)) ./ denom(idx3);
    
    grad_Phi = grad_Phi - matZPs{p}*diagYp*nu_p;
    clear numer_p denom_p idx1 idx2 idx3
    clear vecYp diagYp nu_p
end

grad_Phi = grad_Phi;
% grad_Phi = (1.0/(set.nbL*set.nbP)) * grad_Phi;
grad = grad_Phi + 2*para.gammaC*theta;

end

function lipsc = computeLipscTheta(matZPs, X_inf, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the lipschitz constant
% -------------------------------------------------------------------------

matTemp = zeros(set.nbP, set.nbL);
for p = 1:set.nbP    
    for n = 1:set.nbL
        matTemp(p, n) = norm(matZPs{p}(:, n)*matZPs{p}(:, n)', 2);
        matTemp(p, n) = matTemp(p, n) / X_inf(n);
    end
end
lipsc = (set.nbP*set.nbL*1.0) / sigma * max(matTemp(:)) + 2*para.gammaC;
% lipsc = 1.0 / sigma * max(matTemp(:)) + 2*para.gammaC;
clear matTemp

end

