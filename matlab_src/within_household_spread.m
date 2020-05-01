function [Q_int,states,inf_event_row,inf_event_col] = within_household_spread(composition,sus,det,tau,K_home,alpha,gamma)
% Assuming frequency-dependent homogeneous within-household mixing
% composition(i) isnumber of age class i individuals in the household

hh_size = sum(composition);

classes_present = find(composition>0); % Set of individuals actually present here

K_home = K_home(classes_present,classes_present);
sus = sus(classes_present); det=det(classes_present); tau=tau(classes_present);
r_home = diag(sus)*K_home;

system_sizes = zeros(length(classes_present),1);

for i=1:length(classes_present)
    system_sizes(i) = nchoosek(composition(classes_present(i))+5-1,5-1);
end

total_size = prod(system_sizes);

states = zeros(total_size,5*length(classes_present));

consecutive_repeats = [1; cumprod(system_sizes(1:end-1))]; % Number of times you repeat states for each configuration
block_size = consecutive_repeats.*system_sizes;
num_blocks = total_size./block_size;

for i = 1:length(classes_present)
    k=1;
    for s = 0:composition(classes_present(i))
        for e = 0:composition(classes_present(i))-s
            for d = 0:composition(classes_present(i))-s-e
                for u = 0:composition(classes_present(i))-s-e-d
                    for block=0:num_blocks(i)-1
                       repeat_range = block*block_size(i)+(k-1)*consecutive_repeats(i)+1:block*block_size(i)+k*consecutive_repeats(i);
                       states(repeat_range,5*(i-1)+1:5*i)=ones(consecutive_repeats(i),1)*[s e d u composition(classes_present(i))-s-e-d-u];
                    end
                    k=k+1;
                end
            end
        end
    end    
end

Q_int=sparse(total_size,total_size);

d_pos = 3+5*(0:length(classes_present)-1);
u_pos = 4+5*(0:length(classes_present)-1);

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
index_vector = index_vector+sparse(states(k,:)*reverse_prod'+states(k,end),1,k,state_sizes(1)*reverse_prod(1),1);
end

inf_event_row = [];
inf_event_col = [];

% Add events for each age class
for i = 1:length(classes_present)
    
    s_present = find(states(:,5*(i-1)+1)>0);
    e_present = find(states(:,5*(i-1)+2)>0);
    d_present = find(states(:,5*(i-1)+3)>0);
    u_present = find(states(:,5*(i-1)+4)>0);
    
    % First do infection events
    inf_to = zeros(length(s_present),1);
    inf_rate = zeros(length(s_present),1);
    for k=1:length(s_present)
        old_state = states(s_present(k),:);
        inf_rate(k) = old_state(5*(i-1)+1)*(r_home(i,:)*((old_state(d_pos)./composition(classes_present))'+(old_state(u_pos)./composition(classes_present))'.*tau));
        new_state = old_state;
        new_state(5*(i-1)+1)=new_state(5*(i-1)+1)-1;
        new_state(5*(i-1)+2)=new_state(5*(i-1)+2)+1;
        inf_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
    end
    Q_int = Q_int + sparse(s_present,inf_to,inf_rate,total_size,total_size);
    inf_event_row = [inf_event_row; s_present];
    inf_event_col = [inf_event_col; inf_to];
%     disp('Infection events done');
    % Now do exposure to detected or undetected
    det_to = zeros(length(e_present),1);
    det_rate = zeros(length(e_present),1);
    undet_to = zeros(length(e_present),1);
    undet_rate = zeros(length(e_present),1);
    for k=1:length(e_present)
        % First do detected
        old_state = states(e_present(k),:);
        det_rate(k) = det(i)*alpha*old_state(5*(i-1)+2);
        new_state = old_state;
        new_state(5*(i-1)+2)=new_state(5*(i-1)+2)-1;
        new_state(5*(i-1)+3)=new_state(5*(i-1)+3)+1;
        det_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
        
        % First do undetectednt(k),:);
        undet_rate(k) = (1-det(i))*alpha*old_state(5*(i-1)+2);
        new_state = old_state;
        new_state(5*(i-1)+2)=new_state(5*(i-1)+2)-1;
        new_state(5*(i-1)+4)=new_state(5*(i-1)+4)+1;
        undet_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
    end
    Q_int = Q_int + sparse(e_present,det_to,det_rate,total_size,total_size);
    Q_int = Q_int + sparse(e_present,undet_to,undet_rate,total_size,total_size);
%     disp('Incubaion events done');
    
    % Now do recovery of detected cases
    rec_to = zeros(length(d_present),1);
    rec_rate = zeros(length(d_present),1);
    for k=1:length(d_present)
        old_state = states(d_present(k),:);
        rec_rate(k) = gamma*old_state(5*(i-1)+3);
        new_state = old_state;
        new_state(5*(i-1)+3)=new_state(5*(i-1)+3)-1;
        new_state(5*(i-1)+5)=new_state(5*(i-1)+5)+1;
        rec_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
    end
    Q_int = Q_int + sparse(d_present,rec_to,rec_rate,total_size,total_size);
%     disp('Recovery events from detecteds done');
    
    % Now do recovery of undetected cases
    rec_to = zeros(length(u_present),1);
    rec_rate = zeros(length(u_present),1);
    for k=1:length(u_present)
        old_state = states(u_present(k),:);
        rec_rate(k) = gamma*old_state(5*(i-1)+4);
        new_state = old_state;
        new_state(5*(i-1)+4)=new_state(5*(i-1)+4)-1;
        new_state(5*(i-1)+5)=new_state(5*(i-1)+5)+1;
        rec_to(k) = index_vector(new_state*reverse_prod'+new_state(end));
    end
    Q_int = Q_int + sparse(u_present,rec_to,rec_rate,total_size,total_size); 
%     disp('Recovery events from undetecteds done');   
end

S = sum(Q_int,2);

Q_int = Q_int+sparse(1:total_size,1:total_size,-S);

end