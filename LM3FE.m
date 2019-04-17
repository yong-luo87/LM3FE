function [matUs_opt, theta_opt, matW_opt] = LM3FE(singleTrainFeaL, trainLabelsL, set, para, option)
% -------------------------------------------------------------------------
% Implementation of the large margin multi-view multi-task feature extraction
% -------------------------------------------------------------------------

% -----------------------------------------------------------
% Pre-calculate ||x_n||_inf
% -----------------------------------------------------------
trainFeaL = [];
for v = 1:set.nbV
    trainFeaL = [trainFeaL singleTrainFeaL{v}];
end
trainFeaL_inf = max(abs(trainFeaL), [], 2); clear trainFeaL

% -----------------------------------------------------------
% Initialization of the multi-view combination coefficients
% -----------------------------------------------------------
if option.selfDefinedTheta || option.uniformTheta
    theta = para.theta;
else
    theta = (1.0 / set.nbV) * ones(set.nbV, 1);
end

% -----------------------------------------------------------
% Initialization of the feature selection matrices
% -----------------------------------------------------------
matUs = cell(set.nbV, 1);
for v = 1:set.nbV
    rand('seed', v);
    matUs{v} = rand(set.feaDim(v), set.nbP);
end

% -----------------------------------------------------------
% Initialization of the classification matrix
% -----------------------------------------------------------
matW0 = zeros(set.nbP, set.nbP); vecB0 = zeros(1,set.nbP);
[matW, vecB, obj_W] = optimizeW(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matUs, theta, matW0, vecB0, set, para);
obj = computeObjPreW(obj_W, matUs, theta, set, para);
% obj_temp1 = computeObj(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs, theta, set, para);
obj_ini = obj;

% -------------------------------------------------------------------------
% Solve the optimization problem in an alternative procedure
% -------------------------------------------------------------------------

loop = 1; iter = 0;
while loop
    iter = iter + 1;
    
    % -----------------------------------------------------------
    % Optimize each U{v}
    % -----------------------------------------------------------
    % matUs_new = matUs;
    matUs_new = optimizeUs(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, theta, matUs, set, para, option);
    % obj_temp2 = computeObj(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs_new, theta, set, para);
    clear matUs
    
    % -----------------------------------------------------------
    % Optimize theta if needed
    % -----------------------------------------------------------
    if option.selfDefinedTheta == 0 && option.uniformTheta == 0
        % theta_new = optimizeThetaPGM(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs_new, theta, set, para);
        % theta_new = optimizeThetaOGM(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs_new, theta, set, para);
        theta_new = optimizeTheta(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs_new, theta, set, para);
    else
        theta_new = theta;
    end
    clear matZs
    % obj_temp3 = computeObj(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW, vecB, matUs_new, theta_new, set, para);
    
    % -----------------------------------------------------------
    % Optimize the expansion coefficents
    % -----------------------------------------------------------
    [matW_new, vecB_new, obj_W_new] = ...
        optimizeW(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matUs_new, theta_new, matW, vecB, set, para);
    % obj_temp4 = computeObj(singleTrainFeaL, trainLabelsL, trainFeaL_inf, matW_new, vecB_new, matUs_new, theta_new, set, para);
    
    % -----------------------------------------------------------
    % Compute the objective value
    % -----------------------------------------------------------
    obj_new = computeObjPreW(obj_W_new, matUs_new, theta_new, set, para);
    
    % -----------------------------------------------------------
    % Check the convergence
    % -----------------------------------------------------------
    loop = checkConvergence(obj_new, obj, obj_ini, theta_new, theta, iter, set, para, option);
    
    % -----------------------------------------------------------
    % Update the variables
    % -----------------------------------------------------------
    if loop
        matUs = cell(set.nbV, 1);
        for v = 1:set.nbV
            matUs{v} = matUs_new{v};
        end
        theta = theta_new;
        matW = matW_new;
        vecB = vecB_new;
        obj = obj_new;
        clear matUs_new theta_new matW_new vecB_new obj_new
    end
end
matUs_opt = cell(set.nbV, 1);
for v = 1:set.nbV
    matUs_opt{v} = matUs_new{v};
end
theta_opt = theta_new;
matW_opt = matW_new;

end

