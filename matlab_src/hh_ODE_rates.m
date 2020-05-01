function [dH] = hh_ODE_rates(t,H,Q_int,states,composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix,row,col,total_size)
%hh_ODE_rates calculates the rates of the ODE system describing the
%household ODE model

[FOI_det,FOI_undet] = get_FOI_by_class(H',composition_by_state,states_sus_only,states_det_only,states_undet_only,det_trans_matrix,undet_trans_matrix);
[Q_ext_det, Q_ext_undet] = build_external_import_matrix(states,row,col,FOI_det,FOI_undet,total_size);

dH = (H'*(Q_int+Q_ext_det+Q_ext_undet))';

end

