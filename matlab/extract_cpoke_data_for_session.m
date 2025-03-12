%% Local Function: extract_cpoke_data_for_session
% extract_cpoke_data_for_session(fullpath, animal_id, session_date)

fstring = 'FixationGrower_JessB_R054_240813a';

% Extract animal_id (R049) using string index
animal_id = fstring(22:25);

% Extract date (240811) using string index
date_str = fstring(27:32);

% Construct the full path
fullpath = ['/Volumes/brody/RATTER/SoloData/Data/JessB/', animal_id, ...
             '/data_@', fstring, '.mat'];
session_date =datetime(date_str, 'InputFormat', 'yyMMdd');
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
if (stage < 5 || stage > 11)
    fprintf('Skipping file %s: stage out of range (stage = %d).\n', fullpath, stage);
    cpoke_data = [];
    return;
end

% Initialize the cpoke_data structure.
cpoke_data = struct('sessid', {}, 'animal_id', {}, 'trial', {}, ...
                    'cpoke_dur', {}, 'cpoke_iti', {}, 'post_settling_violation', {}, ...
                    'settling_violation', {}, 'was_rewarded', {}, 'fixation_dur', {},...
                    'date', {}, 'stage', {});

for trial = 1:length(peh)
    try
        if trial == 77
            disp('');
        end
        parsed_events = peh{trial};

        % Skip if the 'cpoke' field is missing or empty.
        if ~isfield(parsed_events.states, 'cpoke') || isempty(parsed_events.states.cpoke)
            last_cpoke_time = 0;
            continue;
        end

        settling_ins = parsed_events.states.settling_in;
        n_settling_ins = size(settling_ins, 1);
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
                    'fixation_dur', fixation_dur, 'date', session_date, ...
                    'stage', stage);
            end
        end

        % Process the valid cpoke (the last settling_in).
        valid_c_in = settling_ins(end, 1);
        c_outs = parsed_events.pokes.C(:,2);
        c_outs = c_outs(c_outs > valid_c_in);

        if ~was_post_settling_violation
            % Choose the first c_out after go_time.
            try
                go_time = parsed_events.states.go_state(1);
            catch
                go_time = NaN; % 1 animal, trial there was a missing go
            end
            c_out = min(c_outs(c_outs >= go_time));
            if isempty(c_out) 
                c_out = NaN; % animal didn't answer (or machine got stuck)
            end
            post_settling_violation = false;
        else
            % Choose the last c_out before the violation time
            if ~isempty(parsed_events.states.violation_due_to_concurrent_spoke)
                c_out=NaN;
            else
                violation_time = parsed_events.states.violation_state(1);
                c_out = max(c_outs(c_outs < violation_time));
            end
            post_settling_violation = true;
        end

        cpoke_dur = c_out - valid_c_in;
        if last_cpoke_time == 0
            cpoke_iti = NaN;
        else
            cpoke_iti = valid_c_in - last_cpoke_time;
        end
        if isnan(c_out)
            last_cpoke_time = valid_c_in;
        else
            last_cpoke_time = c_out;
        end
        was_rewarded = ~isempty(parsed_events.states.hit_state);

        % Append the valid cpoke event.
        cpoke_data(end+1) = struct('sessid', sess_id, 'animal_id', animal_id, ...
                    'trial', trial, 'cpoke_dur', cpoke_dur, 'cpoke_iti', cpoke_iti, ...
                    'post_settling_violation', post_settling_violation, ...
                    'settling_violation', false, 'was_rewarded', was_rewarded, ...
                    'fixation_dur', fixation_dur, 'date', session_date, ...
                    'stage', stage);
    catch ME_trial
        % Report error with file name, session id, and trial number.
        [~,fname,~] = fileparts(fullpath);
        fprintf('Error in file %s, session %s, trial %d: %s\n', ...
            fname, num2str(sess_id), trial, ME_trial.message);
        % Continue processing remaining trials.
    end
end
