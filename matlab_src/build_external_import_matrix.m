function [Q_ext_d, Q_ext_u] = build_external_import_matrix(states,row,col,FOI_det,FOI_undet,total_size)
% Gets sparse matrices containing rates of external infection in a household
% of a given type

d_vals = zeros(1,length(row));
u_vals = zeros(1,length(row));

for i=1:length(row)
    old_state = states(row(i),:);
    new_state = states(col(i),:);
    class_infected = find(new_state(1:5:end)<old_state(1:5:end)); % Figure out which class gets infected in this transition
    d_vals(i) = FOI_det(row(i),class_infected);
    u_vals(i) = FOI_undet(row(i),class_infected);
end

Q_ext_d = sparse(row,col,d_vals,total_size,total_size);
Q_ext_u = sparse(row,col,u_vals,total_size,total_size);

S = sum(Q_ext_d,2);
Q_ext_d = Q_ext_d+sparse(1:total_size,1:total_size,-S);
S = sum(Q_ext_u,2);
Q_ext_u = Q_ext_u+sparse(1:total_size,1:total_size,-S);

end

