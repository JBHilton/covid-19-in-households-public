% This script constructs the internal transmission matrix for a UK-like
% population and a single instance of the external importation matrix.

k_home = readmatrix('../inputs/MUestimates_home_2.xlsx','Sheet','United Kingdom of Great Britain');

k_all = readmatrix('../inputs/MUestimates_all_locations_2.xlsx','Sheet','United Kingdom of Great Britain');

fine_bds = 0:5:80;

coarse_bds = [fine_bds(1:6),fine_bds(13:end)];

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

composition_list = readmatrix('../inputs/uk_composition_list.csv'); % List of observed household compositions
comp_dist = readmatrix('../inputs/uk_composition_dist.csv'); % Proportion of households which are in each composition

size_list = sum(composition_list,2);

% With the parameters chosen, we calculate Q_int:
[Q_int,states,which_composition,system_sizes,cum_sizes,inf_event_row,inf_event_col] = build_household_population(composition_list,sigma,det,tau,k_home,alpha,gamma);

total_size = length(which_composition);

% To define external mixing we need to set up the transmission matrices:

det_trans_matrix = diag(sigma)*k_ext; % Scale rows of contact matrix by age-specific susceptibilities
undet_trans_matrix = diag(sigma)*k_ext*diag(tau); % Scale columns by asymptomatic reduction in transmission
composition_by_state = composition_list(which_composition,:); % This stores number in each age class by household
states_sus_only = states(:,1:5:end); % 1:5:end gives columns corresponding to susceptible cases in each age class in each state
s_present = find(sum(states_sus_only,2)>0);

% Our starting state H is the composition distribution with a small amount
% of infection present:
states_det_only = states(:,3:5:end); % 3:5:end gives columns corresponding to detected cases in each age class in each state
states_undet_only = states(:,4:5:end); % 4:5:end gives columns corresponding to undetected cases in each age class in each state
fully_sus = find(sum(states_sus_only,2)==sum(states,2));
i_is_one = find(sum(states_det_only+states_undet_only,2)==1);
H=zeros(1,total_size);
H(i_is_one) = (1e-5)*comp_dist(which_composition(i_is_one)); % Assing probability of 1e-5 to each member of each composition being sole infectious person in hh
H(fully_sus) = (1-1e-5*sum(comp_dist(which_composition(i_is_one))))*comp_dist; % Assign rest of probability to there being no infection in the household

% Calculate force of infection on each state
[FOI_det,FOI_undet] = get_FOI_by_class(H,composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix);
% Now calculate the external infection components of the transmission
% matrix:
[Q_ext_det, Q_ext_undet] = build_external_import_matrix(states,inf_event_row,inf_event_col,FOI_det,FOI_undet,total_size);
