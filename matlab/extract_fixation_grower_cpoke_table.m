%% Parameters
% Define an array of animal IDs (folders) to process.
animals = {'R041'};  % Add additional animal IDs as needed
% animals = {'R041', 'R042', 'R043', 'R045', 'R046', 'R047', 'R048', 'R049', 'R050', ...
%            'R051', 'R052', 'R053', 'R054', 'R055', 'R056', 'R057'};
% % Define the minimum and maximum dates (using yyyy-MM-dd format)
% min_date_str = '2024-08-06';
% max_date_str = '2024-10-17';
% min_date = datetime(min_date_str, 'InputFormat', 'yyyy-MM-dd');
% max_date = datetime(max_date_str, 'InputFormat', 'yyyy-MM-dd');

% Define animal-specific date ranges as a structure.
animal_dates = struct();
animal_dates.R040 = struct('min', '2024-08-01', 'max', '2024-08-13');
animal_dates.R041 = struct('min', '2024-07-30', 'max', '2024-08-09');
animal_dates.R042 = struct('min', '2024-07-30', 'max', '2024-08-10');
animal_dates.R043 = struct('min', '2024-07-30', 'max', '2024-08-05');
animal_dates.R045 = struct('min', '2024-07-30', 'max', '2024-08-09');
animal_dates.R046 = struct('min', '2024-07-30', 'max', '2024-08-11');
animal_dates.R047 = struct('min', '2024-08-01', 'max', '2024-08-27');
animal_dates.R048 = struct('min', '2024-08-04', 'max', '2024-09-15');
animal_dates.R049 = struct('min', '2024-08-07', 'max', '2024-08-17');
animal_dates.R050 = struct('min', '2024-08-06', 'max', '2024-08-27');
animal_dates.R051 = struct('min', '2024-08-02', 'max', '2024-08-09');
animal_dates.R052 = struct('min', '2024-08-01', 'max', '2024-10-03');
animal_dates.R053 = struct('min', '2024-08-04', 'max', '2024-08-22');
animal_dates.R054 = struct('min', '2024-08-06', 'max', '2024-09-04');
animal_dates.R055 = struct('min', '2024-08-03', 'max', '2024-08-11');
animal_dates.R056 = struct('min', '2024-08-04', 'max', '2024-10-04');
animal_dates.R057 = struct('min', '2024-08-05', 'max', '2024-08-11');

% Base data directory (change if needed)
base_dir = '/Volumes/brody/RATTER/SoloData/Data/JessB';
save_dir = '/Volumes/brody/jbreda/FixationGrower_cpoke_table/';

%% Main Loop: Process each animal
for iAnimal = 1:length(animals)
    animal_id = animals{iAnimal};
    data_dir = fullfile(base_dir, animal_id);
    
    % Retrieve the specific date range for the current animal.
    if isfield(animal_dates, animal_id)
        current_dates = animal_dates.(animal_id);
        min_date_str = current_dates.min;
        max_date_str = current_dates.max;
    else
        error('No date range specified for animal %s', animal_id);
    end

    min_date = datetime(min_date_str, 'InputFormat', 'yyyy-MM-dd');
    max_date = datetime(max_date_str, 'InputFormat', 'yyyy-MM-dd');
    
    % Continue with processing...
    files = dir(fullfile(data_dir, 'data_@FixationGrower_*.mat'));
    all_cpoke_table = table(); % initalize table
    
    fprintf('Processing animal: %s (Date range: %s to %s)\n', animal_id, datestr(min_date), datestr(max_date));
    
    % Loop over each file for this animal.
    for iFile = 1:length(files)
        fullpath = fullfile(data_dir, files(iFile).name);
        
        % Check the file's timestamp (using the dir date)
        file_date = datetime(files(iFile).date);
        if file_date < min_date || file_date > max_date
            continue;  % Skip files outside the date range
        end
        
        fprintf('Processing file: %s (Date: %s)\n', files(iFile).name, datestr(file_date));
        
        % Run the cpoke extraction on this session file.
        try
            session_cpoke_data = extract_cpoke_data_for_session(fullpath, animal_id, file_date);
            % If extraction returned empty (due to stage or empty peh), skip it.
            if isempty(session_cpoke_data)
                continue;
            end
            
            % Convert the session data (struct array) to a table and concatenate.
            session_table = struct2table(session_cpoke_data);
            all_cpoke_table = [all_cpoke_table; session_table];  %#ok<AGROW>
        catch ME_file
            % Report errors (file name, session ID, trial, error message)
            fprintf('Error processing file %s: %s\n', files(iFile).name, ME_file.message);
            continue;
        end
    end
    
    % Save the concatenated table for the animal if data exists.
    if ~isempty(all_cpoke_table)
        output_filename = fullfile(save_dir, sprintf('cpoke_data_%s.csv', animal_id));
        writetable(all_cpoke_table, output_filename);
        fprintf('Saved cpoke data for animal %s to %s\n\n', animal_id, output_filename);
    else
        fprintf('No valid session data found for animal %s within the specified date range.\n\n', animal_id);
    end
end

%% Local Function: extract_cpoke_data_for_session
function cpoke_data = extract_cpoke_data_for_session(fullpath, animal_id, session_date)
% extract_cpoke_data_for_session processes a single session file for the
% FixationGrower task.
%
% Inputs:
%   fullpath     - Full path to the .mat session file.
%   animal_id    - Animal ID (e.g., 'R041') used for folder navigation.
%   session_date - The date of the session (from the file timestamp)
%
% Output:
%   cpoke_data - Struct array with fields:
%       sessid, animal_id, trial, cpoke_dur, cpoke_iti,
%       post_settling_violation, settling_violation, was_rewarded,
%       fixation_dur, date
%
% If the session file does not meet criteria (e.g., empty peh or stage not in [5,7]),
% the function prints a message and returns empty.

    % Load the session file. It is assumed that loading creates variables
    % like "saved", "saved_history", etc.
    load(fullpath);  %#ok<LOAD>
    
    % Extract key variables & Check conditions
    % PEH
    peh = saved_history.ProtocolsSection_parsed_events;
    if isempty(peh)
        fprintf('Skipping file %s: empty peh.\n', fullpath);
        cpoke_data = [];
        return;
    end

    % TRIALS, SESSION ID, STAGE
    n_trials = saved.ProtocolsSection_n_done_trials;
    sess_id = saved.FixationGrower_sessid;
    stage = saved.HistorySection_stage_history(end);
    if (stage < 5 || stage > 7)
        fprintf('Skipping file %s: stage out of range (stage = %d).\n', fullpath, stage);
        cpoke_data = [];
        return;
    end

    % Initialize the cpoke_data structure.
    cpoke_data = struct('sessid', {}, 'animal_id', {}, 'trial', {}, ...
                        'cpoke_dur', {}, 'cpoke_iti', {}, 'post_settling_violation', {}, ...
                        'settling_violation', {}, 'was_rewarded', {}, 'fixation_dur', {},...
                        'date', {});

    for trial = 1:n_trials
        try
            parsed_events = peh{trial};

            % Skip if the 'cpoke' field is missing or empty.
            if ~isfield(parsed_events.states, 'cpoke') || isempty(parsed_events.states.cpoke)
                continue;
            end

            settling_ins = parsed_events.states.settling_in;
            n_settling_ins = size(settling_ins, 1);
            go_time = parsed_events.states.go_state(1);
            was_post_settling_violation = ~isempty(parsed_events.states.violation_state);
            fixation_dur = saved_history.ShapingSection_fixation_dur{trial};
            % TODO Get the fixation dur
            if trial == 1
                last_cpoke_time = 0;
            end

            % Process violation cpokes (if more than one settling in)
            if n_settling_ins > 1
                for settle = 1:(n_settling_ins - 1)
                    % Determine the inter-trial interval (ITI)
                    if last_cpoke_time == 0 && settle == 1
                        cpoke_iti = NaN;
                    else
                        cpoke_iti = settling_ins(settle, 1) - last_cpoke_time;
                    end

                    % Compute the cpoke time for this settling period.
                    cpoke_dur = settling_ins(settle, 2) - settling_ins(settle, 1);
                    last_cpoke_time = settling_ins(settle, 2);
                    settling_violation = true;

                    % Append the cpoke event (violation type) to cpoke_data.
                    cpoke_data(end+1) = struct('sessid', sess_id, 'animal_id', animal_id, ...
                        'trial', trial, 'cpoke_dur', cpoke_dur, 'cpoke_iti', cpoke_iti, ...
                        'post_settling_violation', false, ...
                        'settling_violation', settling_violation, 'was_rewarded', false, ...
                        'fixation_dur', fixation_dur, 'date', session_date);
                end
            end

            % Process the valid cpoke (the last settling_in).
            valid_c_in = settling_ins(end, 1);
            c_outs = parsed_events.pokes.C(:,2);
            c_outs = c_outs(c_outs > valid_c_in);

            if ~was_post_settling_violation
                % Choose the first c_out after go_time.
                c_out = min(c_outs(c_outs >= go_time));
                post_settling_violation = false;
            else
                % Choose the last c_out before the violation time.
                violation_time = parsed_events.states.violation_state(1);
                c_out = max(c_outs(c_outs < violation_time));
                post_settling_violation = true;
            end

            cpoke_dur = c_out - valid_c_in;
            if last_cpoke_time == 0
                cpoke_iti = NaN;
            else
                cpoke_iti = valid_c_in - last_cpoke_time;
            end
            last_cpoke_time = c_out;
            was_rewarded = ~isempty(parsed_events.states.hit_state);

            % Append the valid cpoke event.
            cpoke_data(end+1) = struct('sessid', sess_id, 'animal_id', animal_id, ...
                        'trial', trial, 'cpoke_dur', cpoke_dur, 'cpoke_iti', cpoke_iti, ...
                        'post_settling_violation', post_settling_violation, ...
                        'settling_violation', false, 'was_rewarded', was_rewarded, ...
                        'fixation_dur', fixation_dur, 'date', session_date);
        catch ME_trial
            % Report error with file name, session id, and trial number.
            [~,fname,~] = fileparts(fullpath);
            fprintf('Error in file %s, session %s, trial %d: %s\n', ...
                fname, num2str(sess_id), trial, ME_trial.message);
            % Continue processing remaining trials.
        end
    end
end