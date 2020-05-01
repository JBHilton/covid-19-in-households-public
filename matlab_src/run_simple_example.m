% This sets up and runs a simple system which is low-dimensional enough to
% do in Matlab

k_home = readmatrix('../inputs/MUestimates_home_2.xlsx','Sheet','United Kingdom of Great Britain');

k_all = readmatrix('../inputs/MUestimates_all_locations_2.xlsx','Sheet','United Kingdom of Great Britain');

fine_bds = 0:5:80;

coarse_bds = [0 20 80];

pop_pyr = readmatrix('../inputs/United Kingdom-2019.csv');
pop_pyr = pop_pyr(:,2)+pop_pyr(:,3);

k_home = aggregate_contact_matrix( k_home,fine_bds,coarse_bds,pop_pyr );

k_all= aggregate_contact_matrix( k_all,fine_bds,coarse_bds,pop_pyr );

k_ext = k_all-k_home;

rho = readmatrix('../inputs/rho_estimate_cdc.csv'); % This is in ten year blocks

aggregator=zeros(length(fine_bds)-1,1); % This matrix stores where each class in finer structure is in coarser structure
cdc_bds = 0:10:80;
for i=1:length(fine_bds)-1
aggregator(i)=find(cdc_bds>=fine_bds(i+1),1)-1;
end
rho=sparse(1:length(aggregator),ones(1,length(aggregator)),rho(aggregator)); % This is in five year blocks

rho = aggregate_vector_quantities( rho,fine_bds,coarse_bds,pop_pyr );

det = 0.2*ones(length(rho),1);

tau = 0.5*ones(length(rho),1);

sigma = rho./det;

alpha =1/5;

gamma = 1/2;

composition_list = [0,1;0,2;1,1;2,1;1,2;2,2]; comp_dist = [0.1,0.1,0.2,0.2,0.2,0.2]';

[Q_int,states,which_composition,system_sizes,cum_sizes,inf_event_row,inf_event_col] = build_household_population(composition_list,sigma,det,tau,k_home,alpha,gamma);

size_list = sum(composition_list,2);
size_by_state = size_list(which_composition);

det_trans_matrix = diag(sigma)*k_ext; % Scale rows of contact matrix by age-specific susceptibilities
undet_trans_matrix = diag(sigma)*k_ext*diag(tau); % Scale columns by asymptomatic reduction in transmission
composition_by_state = composition_list(which_composition,:); % This stores number in each age class by household
states_sus_only = states(:,1:5:end); % 1:5:end gives columns corresponding to susceptible cases in each age class in each state
s_present = find(sum(states_sus_only,2)>0);

states_det_only = states(:,3:5:end); % 3:5:end gives columns corresponding to detected cases in each age class in each state
states_undet_only = states(:,4:5:end); % 4:5:end gives columns corresponding to undetected cases in each age class in each state

fully_sus = find(sum(states_sus_only,2)==sum(states,2));
i_is_one = find(sum(states_det_only+states_undet_only,2)==1);
total_size = length(which_composition);
H0=zeros(1,total_size);
H0(fully_sus) = comp_dist;
H0(i_is_one) = (1e-5)*comp_dist(which_composition(i_is_one));


tspan = [0 100];

tic
[t,H] = ode45(@(t,p) hh_ODE_rates(t,p,Q_int,states,composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix,inf_event_row,inf_event_col,total_size),tspan,H0);
toc

ave_children_per_household = H0*(sum(states(:,1:5),2));
ave_adults_per_household = H0*(sum(states(:,6:10),2));
Dc = H*states(:,3)/ave_children_per_household;
Uc = H*states(:,4)/ave_children_per_household;
Da = H*states(:,8)/ave_adults_per_household;
Ua = H*states(:,9)/ave_adults_per_household;

D_type_1 = H(:,1:system_sizes(1))*(states(1:system_sizes(1),3)+states(1:system_sizes(1),8))/comp_dist(1);
U_type_1 = H(:,1:system_sizes(1))*(states(1:system_sizes(1),4)+states(1:system_sizes(1),9))/comp_dist(1);
D_type_2 = H(:,cum_sizes(1)+1:cum_sizes(2))*(states(cum_sizes(1)+1:cum_sizes(2),3)+states(cum_sizes(1)+1:cum_sizes(2),8))/comp_dist(2);
U_type_2 = H(:,cum_sizes(1)+1:cum_sizes(2))*(states(cum_sizes(1)+1:cum_sizes(2),4)+states(cum_sizes(1)+1:cum_sizes(2),9))/comp_dist(2);
D_type_3 = H(:,cum_sizes(2)+1:cum_sizes(3))*(states(cum_sizes(2)+1:cum_sizes(3),3)+states(cum_sizes(2)+1:cum_sizes(3),8))/comp_dist(3);
U_type_3 = H(:,cum_sizes(2)+1:cum_sizes(3))*(states(cum_sizes(2)+1:cum_sizes(3),4)+states(cum_sizes(2)+1:cum_sizes(3),9))/comp_dist(3);
D_type_4 = H(:,cum_sizes(3)+1:cum_sizes(4))*(states(cum_sizes(3)+1:cum_sizes(4),3)+states(cum_sizes(3)+1:cum_sizes(4),8))/comp_dist(4);
U_type_4 = H(:,cum_sizes(3)+1:cum_sizes(4))*(states(cum_sizes(3)+1:cum_sizes(4),4)+states(cum_sizes(3)+1:cum_sizes(4),9))/comp_dist(4);
D_type_5 = H(:,cum_sizes(4)+1:cum_sizes(5))*(states(cum_sizes(4)+1:cum_sizes(5),3)+states(cum_sizes(4)+1:cum_sizes(5),8))/comp_dist(5);
U_type_5 = H(:,cum_sizes(4)+1:cum_sizes(5))*(states(cum_sizes(4)+1:cum_sizes(5),4)+states(cum_sizes(4)+1:cum_sizes(5),9))/comp_dist(5);
D_type_6 = H(:,cum_sizes(5)+1:cum_sizes(6))*(states(cum_sizes(5)+1:cum_sizes(6),3)+states(cum_sizes(5)+1:cum_sizes(6),8))/comp_dist(6);
U_type_6 = H(:,cum_sizes(5)+1:cum_sizes(6))*(states(cum_sizes(5)+1:cum_sizes(6),4)+states(cum_sizes(5)+1:cum_sizes(6),9))/comp_dist(6);

figure
subplot(1,2,1)
set(gca,'DefaultLineLineWidth',3)
plot(t,Dc,'b-',t,Uc,'b:',t,Da,'r-',t,Ua,'r:')
legend('Detected cases in children','Undetected cases in children','Detected cases in adults','Undetected cases in adults');
xlabel('Time in days')
ylabel('Prevalence in age class')
axis square;
set(gca,'FontSize',16);


subplot(1,2,2)
set(gca,'DefaultLineLineWidth',3)
plot(t,D_type_1,'b-',t,U_type_1,'b:',t,D_type_2,'r-',t,U_type_2,'r:',t,D_type_3,'g-',t,U_type_3,'g:',t,D_type_4,'m-',t,U_type_4,'m:',t,D_type_5,'c-',t,U_type_5,'c:',t,D_type_6,'y-',t,U_type_6,'y:')
legend('Detected in 1 adult HH','Undetected in 1 adult HH','Detected in 2 adult HH','Undetected in 2 adult HH','Detected in 1 adult 1 child HH','Undetected in 1 adult 1 child HH','Detected in 1 adult 2 children HH','Undetected in 1 adult 2 children HH','Detected in 2 adult 1 child HH','Undetected in 2 adult 1 child HH','Detected in 2 adult 2 children HH','Undetected in 2 adult 2 children HH');
xlabel('Time in days')
ylabel('Prevalence in household')
axis square;
set(gca,'FontSize',16);
