function [FOI_det,FOI_undet] = get_FOI_by_class(H,composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix)
% H is distribution of states by household
    
    det_by_class = (H*states_det_only)./(H*composition_by_state); % Average detected infected by household in each class
    undet_by_class = (H*states_undet_only)./(H*composition_by_state); % Average undetected infected by household in each class

    FOI_det = states_sus_only*diag(det_trans_matrix*det_by_class'); % This stores the rates of generating an infected of each class in each state
    FOI_undet = states_sus_only*diag(undet_trans_matrix*undet_by_class'); % This stores the rates of generating an infected of each class in each state
    
    
end