function obj = computeObjPreW(obj_W, matUs, theta, set, para)
% -------------------------------------------------------------------------
% Calculate the objective of the optimization problem with the pre-computed
% obj_W
% -------------------------------------------------------------------------

obj_U = 0.0;
for v = 1:set.nbV
    obj_U = obj_U + norm21(matUs{v});
end

obj = obj_W + para.gammaB * obj_U + para.gammaC * (theta'*theta);

end

