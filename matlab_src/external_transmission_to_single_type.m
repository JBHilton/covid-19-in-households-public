function [Q_ext_d, Q_ext_u] = external_transmission_to_single_type(composition,states,FOI_det,FOI_undet)
% Gets sparse matrices containing rates of external infection in a household
% of a given type

classes_present = find(composition>0); % Set of individuals actually present here

system_sizes = zeros(length(classes_present),1);

for i=1:length(classes_present)
    system_sizes(i) = nchoosek(composition(classes_present(i))+5-1,5-1);
end

total_size = prod(system_sizes);

Q_ext_d=sparse(total_size,total_size);
Q_ext_u=sparse(total_size,total_size);

states_refined = zeros(total_size,5*length(classes_present));
for i=1:length(classes_present)
    states_refined(:,5*(i-1)+1:5*i) = states(:,5*(classes_present(i)-1)+1:5*classes_present(i));
end

% Now construct a sparse vector which tells you which row a state appears
% from in the state array

% This loop tells us how many values each column of the state array can
% take
state_sizes=[];
for c=classes_present
    state_sizes=[state_sizes (composition(c)+1)*ones(1,5)];
end

% This vector stores the number of combinations you can get of all
% subsequent elements in the state array, i.e. reverse_prod(i) tells you
% how many arrangements you can get in states(:,i+1:end)
reverse_prod = [0 cumprod(state_sizes(end:-1:2))]; reverse_prod = reverse_prod(end:-1:1);

% We can then define index_vector look up the location of a state by
% weighting its elements using reverse_prod - this gives a unique mapping
% from the set of states to the integers. Because lots of combinations
% don't actually appear in the states array, we use a sparse array which
% will be much bigger than we actually require
index_vector = sparse(state_sizes(1)*reverse_prod(1),1);
for k=1:total_size
index_vector = index_vector+sparse(states_refined(k,:)*reverse_prod'+states_refined(k,end),1,k,state_sizes(1)*reverse_prod(1),1);
end

% Add events for each age class
for i = 1:length(classes_present)
    
    s_present = find(states_refined(:,5*(i-1)+1)>0);
    
    % First do infection events
    inf_to = zeros(length(s_present),1);
    for k=1:length(s_present)
        old_state = states_refined(s_present(k),:);
        new_state = old_state;
        new_state(5*(i-1)+1)=new_state(5*(i-1)+1)-1;
        new_state(5*(i-1)+2)=new_state(5*(i-1)+2)+1;
        inf_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
    end
    Q_ext_d = Q_ext_d + sparse(s_present,inf_to,FOI_det(s_present,classes_present(i)),total_size,total_size);
    Q_ext_u = Q_ext_u + sparse(s_present,inf_to,FOI_undet(s_present,classes_present(i)),total_size,total_size);
 
end

S = sum(Q_ext_d,2);
Q_ext_d = Q_ext_d+sparse(1:total_size,1:total_size,-S);
S = sum(Q_ext_u,2);
Q_ext_u = Q_ext_u+sparse(1:total_size,1:total_size,-S);

end
