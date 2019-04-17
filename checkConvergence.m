function loop = checkConvergence(obj_new, obj, obj_ini, theta_new, theta, iter, set, para, option)
% -------------------------------------------------------------------------
% Check if the stop criterion is reached
% -------------------------------------------------------------------------

loop = 1;
if abs(obj_ini - obj_new) > eps
    obj_diff = abs(obj - obj_new) / abs(obj_ini - obj_new);
else
    obj_diff = abs(obj - obj_new);
end

% -----------------------------------------------------------
% Verbosity
% -----------------------------------------------------------
if option.verbose >= 1
    if iter == 1 || rem(iter,10) == 0
        fprintf('-------------------------------------------------\n');
        fprintf('Iter | Obj.    | Obj_new  | Obj_diff  | DiffThetas |\n');
        fprintf('-------------------------------------------------\n');
    end;
    fprintf('%d    |%8.4f | %8.4f | %8.4f  | %6.4f   \n', [iter obj obj_new obj_diff max(abs(theta_new-theta))]);
end
if option.verbose >= 2
    fprintf('theta_new = ');
    for iv = 1:set.nbV
        fprintf('%.4f ', theta_new(iv));
    end
    fprintf('\n');    
end

% -----------------------------------------------------------
% Check difference of obj. value conditions
% -----------------------------------------------------------
if option.stopdiffobj == 1 && obj_diff < para.seuildiffobj
    loop = 0;
    fprintf(1,'obj. difference convergence criteria reached \n');
end

% -----------------------------------------------------------
% Check variation of theta conditions
% -----------------------------------------------------------
if  option.stopvariationtheta == 1 && max(abs(theta_new-theta)) < para.seuildifftheta
    loop = 0;
    fprintf(1,'theta variation convergence criteria reached \n');
end

% -----------------------------------------------------------
% Check number of iteration conditions
% -----------------------------------------------------------
if iter >= para.nbIterMax
    loop = 0;
    fprintf(1,'maximum number of iterations reached \n');
end

end

