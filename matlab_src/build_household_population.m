function [Q_int,states,which_composition,system_sizes,cum_sizes,inf_event_row,inf_event_col] = build_household_population(composition_list,sus,det,tau,K_home,alpha,gamma)
% This builds internal mixing matrix for entire system of age-structured
% households
    % If the compositions include household size at the beginning, we throw
    % it away here. While we would expect to see some households with equal
    % numbers in age class 1 and all others combined, we should not see it
    % everywhere and so this is a safe way to check.
    
    if max(abs(composition_list(:,1)-sum(composition_list(:,2:end),2)))==0
        size_list = composition_list(:,1);
        composition_list = composition_list(:,2:end);
    else
        size_list = sum(composition_list,2);
    end
    
    [no_types, no_classes] = size(composition_list);
    
    classes_present = composition_list>0; % This is an array of logicals telling you which classes are present in each composition
    
    system_sizes = ones(no_types,1);
    for i=1:no_types
        for j=find(classes_present(i,:))
            system_sizes(i) = system_sizes(i)*nchoosek(composition_list(i,j)+5-1,5-1);
        end
    end
    cum_sizes = cumsum(system_sizes); % This is useful for placing blocks of system states
    total_size = cum_sizes(end); % Total size is sum of the sizes of each composition-system, because we are considering a single household which can be in any one composition
    states = zeros(total_size,5*no_classes); % Stores list of (S,E,D,U,R)_a states for each composition
    which_composition = zeros(total_size,1);

    % Initialise matrix of internal process by doing first block
    which_composition(1:system_sizes(1)) = ones(system_sizes(1),1);
    [Q_temp,states_temp,inf_event_row,inf_event_col] = within_household_spread(composition_list(1,:),sus,det,tau,K_home,alpha,gamma);
    Q_int = sparse(Q_temp);
    class_list = find(classes_present(1,:));
    for j=1:length(class_list)
        this_class = class_list(j);
        states(1:system_sizes(1),5*(this_class-1)+1:5*this_class) = states_temp(:,5*(j-1)+1:5*j);
    end
    
    % NOTE: The way I do this loop is very wasteful, I'm making lots of
    % arrays which I'm overwriting with different sizes
    start_time=cputime;
    matrix_sizes = system_sizes.^2; % Just store this so we can estimate remaining time
    for i=2:no_types
        which_composition(cum_sizes(i-1)+1:cum_sizes(i)) = i*ones(system_sizes(i),1);
        [Q_temp,states_temp,inf_row_temp,inf_col_temp] = within_household_spread(composition_list(i,:),sus,det,tau,K_home,alpha,gamma);
        Q_int = blkdiag(Q_int,sparse(Q_temp));
        class_list = find(classes_present(i,:));
        for j=1:length(class_list)
            this_class = class_list(j);
            states(cum_sizes(i-1)+1:cum_sizes(i),5*(this_class-1)+1:5*this_class) = states_temp(:,5*(j-1)+1:5*j);
        end
        inf_event_row = [inf_event_row; cum_sizes(i-1)+inf_row_temp];
        inf_event_col = [inf_event_col; cum_sizes(i-1)+inf_col_temp];
        disp([num2str(i) ' blocks calculated. ' num2str(cputime-start_time) ' seconds elapsed, approximately ' num2str(((cputime-start_time)/sum(system_sizes(1:i)))*sum(system_sizes(i+1:end))) ' seconds remaining.']);
    end
    
end