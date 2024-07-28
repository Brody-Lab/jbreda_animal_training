% saved_history.ProtocolsSection_parsed_events{1,1}.states.cpoke(1)
% saved_history.ProtocolsSection_parsed_events{1,1}.states.violation_state(1)
n_trials = saved.ProtocolsSection_n_done_trials;
% 
% 
% saved_history.ShapingSection_delay_dur{1} + saved_history.ShapingSection_stimulus_dur{1} * 2 + saved_history.ShapingSection_pre_dur{1} + saved_history.ShapingSection_post_dur{1}
% saved_history.ShapingSection_settling_in_dur{1}
% 
fix_s = colvec(saved.HistorySection_fixation_history);

for itrial = 1 : n_trials
    
    if ~isempty(saved_history.ProtocolsSection_parsed_events{itrial,1}.states.violation_state);
        viol_time = saved_history.ProtocolsSection_parsed_events{itrial,1}.states.violation_state(1);
        t_start = saved_history.ProtocolsSection_parsed_events{itrial,1}.states.cpoke(1) + saved_history.ShapingSection_settling_in_dur{itrial};
        
        valid_time = viol_time - t_start;
        
        time_left = fix_s(itrial) - valid_time;
        
        fprintf('\ntrial %s: %f seconds left', num2str(itrial), time_left); 
    else
        sprintf('\ntrial %s: valid', num2str(itrial));
        
    end
end