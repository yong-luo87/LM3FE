function w = simplexPro(v, z)
% -------------------------------------------------------------------------
% Efficient projection onto the simplex
% -------------------------------------------------------------------------

v_sort = sort(v, 'descend');

for j = length(v):-1:1
    if v_sort(j) - (1.0/j)*(sum(v_sort(1:j)) - z) > 0
        break;
    end
end

theta = (1.0/j)*(sum(v_sort(1:j)) - z);

w = max((v-theta), 0);

end

