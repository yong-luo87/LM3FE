function obj = computeObj(matXs, matY, X_inf, matW, vecB, matUs, theta, set, para)
% -------------------------------------------------------------------------
% Calculate the objective of the optimization problem
% -------------------------------------------------------------------------

sigma = para.sigma;

matZ = zeros(set.nbP, set.nbL);
for v = 1:set.nbV
    matZ = matZ + theta(v) * matUs{v}' * matXs{v}';
end
matZe = [matZ; ones(1, set.nbL)];

denom = sigma*X_inf;
obj_Phi_temp = zeros(set.nbL, set.nbP);
for p = 1:set.nbP
    vecYp = matY(:,p); diagYp = diag(vecYp);
    vecWpBp = [matW(:,p); vecB(p)];
    numer_p = diagYp * (matZe' * vecWpBp);
    
    idx1 = find(numer_p > 1);
    idx2 = find(numer_p < (1-denom));
    idx3 = setdiff((1:set.nbL)', [idx1; idx2]);
    
    obj_Phi_temp(idx2, p) = (1 - numer_p(idx2)) - 0.5*denom(idx2);
    obj_Phi_temp(idx3, p) = (1 - numer_p(idx3)).^2 ./ (2.0*denom(idx3));
    clear numer_p idx1 idx2 idx3
    clear vecYp diagYp vecWpBp
end
obj_Phi = sum(obj_Phi_temp(:)); clear obj_Phi_temp
% obj_Phi = (1.0/(set.nbL*set.nbP)) * sum(obj_Phi_temp(:)); clear obj_Phi_temp
obj_W = obj_Phi + para.gammaA*norm(matW, 'fro')^2; clear obj_Phi

obj_U = 0.0;
for v = 1:set.nbV
    obj_U = obj_U + norm21(matUs{v});
end

obj = obj_W + para.gammaB * obj_U + para.gammaC * (theta'*theta);

end

