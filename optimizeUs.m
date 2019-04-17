function matUs_new = optimizeUs(matXs, matY, X_inf, matW, vecB, theta, matUs, set, para, option)
% -------------------------------------------------------------------------
% Optimization of the feature selection matrices U{v}
% -------------------------------------------------------------------------

maxit = 1; % 10;
epsilon = 1e-2;
sigma = para.sigma;

lipsc_const = zeros(set.nbV, 1);
for v = 1:set.nbV
    matTemp = zeros(set.nbP, set.nbL);
    for p = 1:set.nbP
        for n = 1:set.nbL
            matTemp(p, n) = norm(matXs{v}(n,:)'*matW(:,p)', 2) * norm(matXs{v}(n,:)', 2) * norm(matW(:,p), 2);
            matTemp(p, n) = matTemp(p, n) / X_inf(n);
        end
    end
    lipsc_const(v) = (set.nbP*set.nbL*theta(v)^2) / sigma * max(matTemp(:)); clear matTemp
    % lipsc_const(v) = theta(v)^2 / sigma * max(matTemp(:)); clear matTemp
end

% -------------------------------------------------------------------------
% Optimize U{v} alternatively until convergence
% -------------------------------------------------------------------------

tic; fprintf('Optimizing U{v} ... ');
if option.verbose >= 3, fprintf('\n'); else fprintf('iter '); end

matUs_new = cell(set.nbV, 1);
% ---------------------------------------------------------
% Initialization of the objective
% ---------------------------------------------------------
obj = computeObj(matXs, matY, X_inf, matW, vecB, matUs, theta, set, para);
obj_ini = obj;

loop = 1; iter = 0;
while loop
    iter = iter + 1;
    if option.verbose >= 3, fprintf('iter %d: ', iter); else fprintf('%d ', iter); end
    for v = 1:set.nbV
        if option.verbose >= 3, fprintf('%d ', v); end
        matXv = matXs{v}';
        
        % ---------------------------------------------------------
        % Pre-calculation
        % ---------------------------------------------------------
        matZs = cell(set.nbV, 1);
        for v2 = 1:set.nbV
            if v2 ~= v
                matZs{v2} = theta(v2)*matXs{v2}*matUs{v2};
            end
        end
        matC = zeros(set.nbL, set.nbP);
        for p = 1:set.nbP
            for v2 = 1:set.nbV
                if v2 ~= v
                    matC(:,p) = matC(:,p) + matZs{v2}*matW(:,p);
                end
            end
            matC(:,p) = matC(:,p) + vecB(p);
        end
        clear matZs
        
        % ---------------------------------------------------------
        % Optimize each U{v}
        % ---------------------------------------------------------
        
        % obj_temp_sub = computeObjUv(matXv, matY, X_inf, matW, matC, matUs{v}, theta(v), sigma, set, para);
        
        [matUv_new, obj_Uv] = optimizeUv(matXv, matY, X_inf, matW, matC, matUs{v}, theta(v), lipsc_const(v), set, para);
        
        % obj_temp_sub_new = computeObjUv(matXv, matY, X_inf, matW, matC, matUv_new, theta(v), sigma, set, para);
        
        % obj_Us = 0.0;
        % for v2 = 1:set.nbV
        %     if v2 ~= v
        %         obj_Us = obj_Us + norm21(matUs{v2});
        %     end
        % end
        % obj_temp = obj_Uv + para.gammaB*obj_Us + para.gammaA*norm(matW, 'fro')^2 + para.gammaC*(theta'*theta);
        
        % ---------------------------------------------------------
        % Update the solution
        % ---------------------------------------------------------
        matUs{v} = matUv_new;
        clear matXv matC matUv matUv_guess
        
        % obj_temp2 = computeObj(matXs, matY, X_inf, matW, vecB, matUs, theta, set, para);
    end
    
    % ---------------------------------------------------------
    % Update the objective
    % ---------------------------------------------------------
    obj_new = computeObj(matXs, matY, X_inf, matW, vecB, matUs, theta, set, para);
    
    % ---------------------------------------------------------
    % Check convergence
    % ---------------------------------------------------------
    if abs(obj_ini - obj_new) > eps
        obj_diff = abs(obj - obj_new) / abs(obj_ini - obj_new);
    else
        obj_diff = abs(obj - obj_new);
    end
    if obj_diff < epsilon || iter >= maxit
        loop = 0;
    end
    
    % ---------------------------------------------------------
    % Update the variables
    % ---------------------------------------------------------
    if loop
        clear obj
        obj = obj_new;
        clear obj_new
        
        if option.verbose >= 3 && rem(iter, 4) == 0
            fprintf('\n');
        end
    end
end
for v = 1:set.nbV
    matUs_new{v} = matUs{v};
end
timecost = toc;
fprintf('Finished! timecost = %.4f s \n', timecost);

end



function [matUv_new, obj_new] = optimizeUv(matXv, matY, X_inf, matW, matC, matUv, theta_v, lipsc_const_v, set, para)
% -------------------------------------------------------------------------
% Optimize U{v} using the Nesterov's gradient method
% -------------------------------------------------------------------------

maxit = 100;
epsilon = 1e-3;
sigma = para.sigma;

% matUv = zeros(size(matUv));
% matUv_guess = zeros(size(matUv));
matUv = matUv;
matUv_guess = matUv;

% -----------------------------------------------------------
% Initialization of the objective, gradient and Lipschitz constant
% -----------------------------------------------------------
[obj(1,1), obj_Phi] = computeObjUv(matXv, matY, X_inf, matW, matC, matUv, theta_v, sigma, set, para);
matDv = diag(0.5./sqrt(sum(matUv.^2, 2) + eps));
grad{1,1} = computeGradUv(matXv, matY, X_inf, matW, matC, matUv, matDv, theta_v, sigma, set, para);
lipsc(1,1) = computeLipsc(matDv, lipsc_const_v, para); clear matDv

% -----------------------------------------------------------
% Nesterov's optimal gradient method for optimize U{v}
% -----------------------------------------------------------
loop = 1; t = 1;
while loop
    % ------------------------------------------------------
    % Solving the two auxiliary optimization problems
    % ------------------------------------------------------
    Y = matUv - 1.0/lipsc(t,1)*grad{t,1};
    tempGrad = zeros(size(grad{1,1}));
    for i = 1:t
        tempGrad = tempGrad + (i/2.0)*grad{i,1};
    end
    Z = matUv_guess - 1.0/lipsc(t,1)*tempGrad;
    clear tempGrad
    
    % ------------------------------------------------------
    % Project the result to [0, 1]
    % ------------------------------------------------------
    % z = (z - min(z)) / (max(z) - min(z));
    
    % ------------------------------------------------------
    % Update the solution
    % ------------------------------------------------------
    matUv_new = (2.0/(t+3))*Z + ((t+1)*1.0)/(t+3)*Y; clear Y Z
    
    t = t + 1;
    % sigma = para.sigma / t;
    %     if sigma > 1
    %         sigma = para.sigma / t;
    %     end
    
    % ------------------------------------------------------
    % Update the objective, gradient and Lipschitz constant
    % ------------------------------------------------------
    [obj(t,1), obj_Phi] = computeObjUv(matXv, matY, X_inf, matW, matC, matUv_new, theta_v, sigma, set, para);
    matDv_new = diag(0.5./sqrt(sum(matUv_new.^2, 2) + eps));
    grad{t,1} = computeGradUv(matXv, matY, X_inf, matW, matC, matUv_new, matDv_new, theta_v, sigma, set, para);
    lipsc(t,1) = computeLipsc(matDv_new, lipsc_const_v, para); clear matDv_new
    
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
        clear matUv
        matUv = matUv_new;
        clear matUv_new
    end
end
obj_new = obj(t,1);
clear obj grad lipsc

end



function [obj, obj_Phi] = computeObjUv(matXv, matY, X_inf, matW, matC, matUv, theta_v, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the objective value
% -------------------------------------------------------------------------

norm_type = 'L21_norm'; % 'L21_norm' or 'fro_norm'

denom = sigma*X_inf;
obj_Phi_temp = zeros(set.nbL, set.nbP);

for p = 1:set.nbP
    vecYp = matY(:, p); diagYp = diag(vecYp);
    vecWp = matW(:, p); numer_p = matC(:, p);
    numer_p = numer_p + theta_v*matXv'*matUv*vecWp;
    numer_p = diagYp * numer_p;
    
    idx1 = find(numer_p > 1);
    idx2 = find(numer_p < (1-denom));
    idx3 = setdiff((1:set.nbL)', [idx1; idx2]);
    
    obj_Phi_temp(idx2, p) = (1-numer_p(idx2)) - 0.5*denom(idx2);
    obj_Phi_temp(idx3, p) = (1-numer_p(idx3)).^2 ./ (2.0*denom(idx3));
    clear numer_p idx1 idx2 idx3
    clear vecYp diagYp vecWp
end

obj_Phi = sum(obj_Phi_temp(:)); clear obj_Phi_temp
% obj_Phi = (1.0/(set.nbL*set.nbP)) * sum(obj_Phi_temp(:)); clear obj_Phi_temp
if strcmp(norm_type, 'L21_norm') == 1
    obj = obj_Phi + para.gammaB*norm21(matUv);
else
    obj = obj_Phi + para.gammaB*norm(matUv, 'fro')^2;
end

end

function grad = computeGradUv(matXv, matY, X_inf, matW, matC, matUv, matDv, theta_v, sigma, set, para)
% -------------------------------------------------------------------------
% Compute the gradient
% -------------------------------------------------------------------------

norm_type = 'L21_norm'; % 'L21_norm' or 'fro_norm'

denom = sigma*X_inf;
grad_Phi = zeros(size(matUv));

for p = 1:set.nbP
    vecYp = matY(:, p); diagYp = diag(vecYp);
    vecWp = matW(:, p); numer_p = matC(:, p);
    numer_p = numer_p + theta_v*matXv'*matUv*vecWp;
    numer_p = diagYp * numer_p;
    
    idx1 = find(numer_p > 1);
    idx2 = find(numer_p < (1-denom));
    idx3 = setdiff((1:set.nbL)', [idx1; idx2]);
    
    nu_p = zeros(set.nbL, 1);
    nu_p(idx2) = 1;
    nu_p(idx3) = (1 - numer_p(idx3)) ./ denom(idx3);
    
    grad_Phi = grad_Phi - theta_v*matXv*diagYp*nu_p*vecWp';
    clear numer_p idx1 idx2 idx3
    clear vecYp diagYp nu_p vecWp
end

grad_Phi = grad_Phi;
% grad_Phi = (1.0/(set.nbL*set.nbP)) * grad_Phi;
if strcmp(norm_type, 'L21_norm') == 1
    grad = grad_Phi + 2*para.gammaB*matDv*matUv;
else
    grad = grad_Phi + 2*para.gammaB*matUv;
end

end

function lipsc = computeLipsc(matDv, lipsc_const_v, para)
% -------------------------------------------------------------------------
% Compute the lipschitz constant
% -------------------------------------------------------------------------

norm_type = 'L21_norm'; % 'L21_norm' or 'fro_norm'

if strcmp(norm_type, 'L21_norm') == 1
    lipsc = lipsc_const_v + 2*para.gammaB*norm(matDv,2);
else
    lipsc = lipsc_const_v + 2*para.gammaB;
end

end

