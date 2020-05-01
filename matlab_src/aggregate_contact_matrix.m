function [ k_coarse ] = aggregate_contact_matrix( k_fine,fine_bds,coarse_bds,pyramid )
%aggregate_age_structure aggregates an age-structured contact matrice
%to return the corresponding transmission matrix under a finer age
%structure
%   Detailed explanation goes here
aggregator=zeros(length(fine_bds)-1,1); % This matrix stores where each class in finer structure is in coarser structure
for i=1:length(fine_bds)-1
    aggregator(i)=find(coarse_bds>=fine_bds(i+1),1)-1;
end

% The Prem et al. estimates cut off at 80, so we all >75 year olds into one
% class for consistency with these estimates:
pyramid(length(fine_bds)-1) = sum(pyramid(length(fine_bds)-1:end));
pyramid = pyramid(1:length(fine_bds)-1);

pyramid = pyramid/sum(pyramid); % Normalise to give proportions
agg_pop_pyramid=sum(sparse(aggregator,1:length(aggregator),pyramid),2)'; % sparse matrix defines here just splits pyramid into rows corresponding to coarse boundaries, then summing each row gives aggregated pyramid

rel_weights = pyramid'./agg_pop_pyramid(aggregator);

% Now define contact matrix with age classes from li et al data
pop_weight_matrix = sparse(aggregator,1:length(aggregator),rel_weights);
pop_no_weight=sparse(aggregator,1:length(aggregator),ones(1,length(aggregator)));
k_coarse = pop_weight_matrix*k_fine*pop_no_weight';

end

