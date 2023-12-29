% Early Give Dur Pokes
% 2023-12-24 
% Goal: see if R029 who is on the pro-anti with light give delay on anti is
% poking early in the give delay period to the incorrect side at all. 
%
% There will be two interesting metrics:
%  (1) number of trials with early incorrect pokes
%  (2) number of trials with early incorrect pokes that switch to correct
%  pre-give
%

base_path = '/Volumes/brody/RATTER/SoloData/Data/JessB/R029/';
files = {'data_@DMS2_JessB_R029_231222b.mat', ...
        'data_@DMS2_JessB_R029_231221a.mat', ...
        'data_@DMS2_JessB_R029_231220a', ...
        'data_@DMS2_JessB_R029_231219a.mat'};


% Initialize a struct array for all sessions
all_sessions_data = struct('date', {}, 'trial', {}, 'block_type', {},...
                    'side', {}, 'result', {}, 'give_delay', {},...
                    'give_use', {}, 'n_incorrect_pokes', {});

for i = 1:length(files)
    file = files{i};
    full_path = [base_path file];
    date = file(end-10:end-5);
    data = load(full_path);

    % Number of trials
    n_trials = data.saved.ProtocolsSection_n_done_trials;

    for trial = 1:n_trials
        disp(trial)
        block_type = data.saved_history.StimulusSection_pro_anti_block_type(trial);
    
        if strcmp(block_type, 'pro')
            continue; % Skip this trial
        end
    
        peh = data.saved_history.ProtocolsSection_parsed_events{trial};
        give_delay = data.saved_history.ShapingSection_give_del_dur(trial);
        give_use = data.saved.HistorySection_give_use_history(trial);
        side = data.saved_history.SideSection_current_side(trial);
        result = data.saved.HistorySection_result_history(trial);
     
        if strcmp(side, "LEFT")
            incorrect_side = "R";
        else
            incorrect_side = "L";
        end
    
        try
            start_time = peh.states.pre_give_delay(1);
        catch
            continue
        end
        end_time = peh.states.pre_give_delay(2);
    
        incorrect_pokes = peh.pokes.(incorrect_side);
    
        incorrect_during_delay = incorrect_pokes(incorrect_pokes > start_time & incorrect_pokes < end_time);
        if isempty(incorrect_during_delay)
            n_incorrect_pokes = 0;
        else
            n_incorrect_pokes = length(incorrect_during_delay(:,1));
        end

        % Append to struct
        all_sessions_data(end+1) = struct('date', date, 'trial', trial, 'block_type', block_type, 'result', result, ...
                                    'side', side, 'give_use', give_use, 'give_delay', give_delay, ...
                                    'n_incorrect_pokes', n_incorrect_pokes);
    end
end

% Convert struct array to table
all_sessions_data = struct2table(all_sessions_data);

% Save table as CSV
writetable(all_sessions_data, '2023_12_24_R029_give_del_pokes.csv');
