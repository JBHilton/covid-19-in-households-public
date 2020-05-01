% This runs the UK-like model with a single set of parameters for 100 days

k_home = readmatrix('inputs/MUestimates_home_2.xlsx','Sheet','United Kingdom of Great Britain');
k_all = readmatrix('inputs/MUestimates_all_locations_2.xlsx','Sheet','United Kingdom of Great Britain');

fine_bds = 0:5:80;
coarse_bds = [fine_bds(1:6),fine_bds(13:end)];

pop_pyr = readmatrix('inputs/United Kingdom-2019.csv');
pop_pyr = pop_pyr(:,2)+pop_pyr(:,3);

k_home = aggregate_contact_matrix( k_home,fine_bds,coarse_bds,pop_pyr );
k_all= aggregate_contact_matrix( k_all,fine_bds,coarse_bds,pop_pyr );
k_ext = k_all-k_home;

rho = readmatrix('inputs/rho_estimate_cdc.csv'); % This is in ten year blocks

aggregator=zeros(length(fine_bds)-1,1); % This matrix stores where each class in finer structure is in coarser structure
cdc_bds = 0:10:80;
for i=1:length(fine_bds)-1
aggregator(i)=find(cdc_bds>=fine_bds(i+1),1)-1;
end
rho=sparse(1:length(aggregator),ones(1,length(aggregator)),rho(aggregator)); % This is in five year blocks

R0=2.4;
gamma = 1/2;
rho = gamma*R0*aggregate_vector_quantities( rho,fine_bds,coarse_bds,pop_pyr ); % Rescale to appropriate units
det = (0.9/max(rho))*rho; % Set det so max is 90%
sigma = rho./det; % Rho and det define sigma
tau = 0*ones(length(rho),1); % No asymptomatic transmission
alpha =1/5;

composition_list = readmatrix('inputs/uk_composition_list.csv'); % List of observed household compositions
comp_dist = readmatrix('inputs/uk_composition_dist.csv'); % Proportion of households which are in each composition

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

% Our starting state H0 is the composition distribution with a small amount
% of infection present:
states_det_only = states(:,3:5:end); % 3:5:end gives columns corresponding to detected cases in each age class in each state
states_undet_only = states(:,4:5:end); % 4:5:end gives columns corresponding to undetected cases in each age class in each state
fully_sus = find(sum(states_sus_only,2)==sum(states,2));
i_is_one = find(sum(states_det_only+states_undet_only,2)==1);
H0=zeros(1,total_size);
H0(i_is_one) = (1e-5)*comp_dist(which_composition(i_is_one)); % Assing probability of 1e-5 to each member of each composition being sole infectious person in hh
H0(fully_sus) = (1-1e-5*sum(comp_dist(which_composition(i_is_one))))*comp_dist; % Assign rest of probability to there being no infection in the household

tspan = [0 100];
tic
[t,H] = ode45(@(t,p) hh_ODE_rates(t,p,Q_int,states,composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix,inf_event_row,inf_event_col,total_size),tspan,H0);
toc

D = H*states(:,3:5:end);
U = H*states(:,4:5:end);

lgd={};

for l=1:length(coarse_bds)-1
    lgd{l} = ['Age ' num2str(coarse_bds(l)) ' to ' num2str(coarse_bds(l+1))];
end
clist=0.5*ones(10,3);
clist(:,1)=(1/11)*(1:10);
clist(:,3)=(1/11)*(10:-1:1);

figure
subplot(1,2,1)
hold on
for i=1:length(coarse_bds)-1
    plot(t,D(:,i),'LineWidth',2,'Color',clist(i,:));
end
hold off
legend(lgd);
xlabel('Time in days')
ylabel('Detected prevalence')
axis square;
set(gca,'FontSize',16);

subplot(1,2,2)
hold on
for i=1:length(coarse_bds)-1
    plot(t,U(:,i),'LineWidth',2,'Color',clist(i,:));
end
hold off
colormap jet;
legend(lgd);
xlabel('Time in days')
ylabel('Undetected prevalence in age class')
axis square;
set(gca,'FontSize',16);


