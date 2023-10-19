function ratdata = extract_AthenaDelayComp_data(ratname, varargin)

    % stage 1: % data_@AthenaDelayComp_Athena_W075_150716a.mat

    warning('OFF','all')

    %map_bucket_drive;
    dirPath = ['X:\RATTER\SoloData\Data\Athena\', ratname];
    cd(dirPath);
    
    if nargin == 2
        specific_file = varargin{1};
        files = dir(fullfile(dirPath, specific_file));
    else
        files = dir(dirPath);
    end
    
    % TODO- could find a better way to make these values with the
    % fileds_to_update thing I am working in below. could also use this to
    % fill in these fields
    ratdata = {};
    ratdata.rat_name           = {};
    ratdata.session_date       = {};
    ratdata.session_counter    = [];
    ratdata.rig_id             = {};
    
    ratdata.training_stage     = [];
    ratdata.A1_dB              = [];
    ratdata.A2_dB              = [];
    
    session_number = 0;
    for i = 1:numel(files)
        
        % skip any folders and only use .mat files
        if files(i).isdir == 1; continue; end

        % TODO figure out why this is heere and if there is a cleaner way
        % to write it? Might be easier to search for AthenaDelayComp in the
        % list and drop all the others
        brks = find(files(i).name == '_');
        if numel(brks) == 4
            protocol = files(i).name(brks(1)+2:brks(2)-1);
            if strcmp(protocol,'AthenaDelayComp')

                disp(['Loading ',files(i).name]);
                load(files(i).name)

                % skip if in spoke only stage
                peh = saved_history.ProtocolsSection_parsed_events;
                if isfield(peh{1}.states, 'sideled_on')
                    disp(['Skipping ' files(i).name, ' because spoke only stage']);
                    continue
                end

                % skip if 0 trials were done in the session
                n_trials = saved.ProtocolsSection_n_done_trials;
                if n_trials <= 0
                    disp(['Skipping ' files(i).name, ' because n trials == 0'])
                    continue
                else
                    session_number = session_number + 1;
                    session_date = files(i).name(brks(4)+1:brks(4)+6);
                    rig_id = regexp(saved.SavingSection_hostname, '\d+', 'match');
                end


                %% Make n_trial long identifiers
                
                for trial = 1:n_trials
                        ratdata.rat_name{end+1}        = saved.SavingSection_ratname;
                        ratdata.session_date{end+1}    = session_date;
                        ratdata.session_counter(end+1) = session_number;
                        ratdata.rig_id{end+1}          = rig_id;
                end
                
                %% Calcualte & Infer Settings
                % Training Stage
                ratdata.training_stage(end+1:end+n_trials) = getTrainingStage(...
                    saved_history.SideSection_reward_type,...
                    saved_history.SideSection_stimuli_on,...
                    n_trials);

                % Sa/Sb (a1/a2) sigma -> dB
                if max(ratdata.training_stage) > 1
                    ratdata.A1_dB(end+1:end+n_trials) = getSoundDb(...
                        saved_history.StimulusSection_A1_sigma,...
                        n_trials);
                    
                    ratdata.A2_dB(end+1:end+n_trials) = getSoundDb(...
                        saved_history.StimulusSection_A2_sigma,...
                        n_trials);       
                                            
                else
                    ratdata.A1_dB(end+1:end+n_trials) = nan(n_trials, 1);
                    ratdata.A2_dB(end+1:end+n_trials) = nan(n_trials, 1);
                end
                
                
                %% Variables from saved or saved_history

                all_vars = {
                    's', 'AthenaDelayComp_hit_history';
                    's', 'AthenaDelayComp_violation_history';
                    'sh', 'StimulusSection_A1_sigma';
                    'sh', 'StimulusSection_Rule';
                    'sh', 'SideSection_violation_iti';
                    'sh', 'SideSection_error_iti';
                    'sh', 'SideSection_secondhit_delay';
                    'sh', 'SideSection_PreStim_time';
                    'sh', 'SideSection_A1_time';
                    'sh', 'SideSection_Del_time';
                    'sh', 'SideSection_A2_time';
                    'sh', 'SideSection_time_bet_aud2_gocue';
                    'sh', 'SideSection_time_go_cue';
                    'sh', 'SideSection_CP_duration'
                    'sh', 'WaterValvesSection_Left_volume';
                    'sh', 'WaterValvesSection_Right_volume';
                    'sh', 'AntibiasSectionAthena_Beta';
                    'sh', 'StimulusSection_psych_pairs';
                };
                

                % Iterate through all variables
                for i = 1:size(all_vars, 1)
                    src_struct = all_vars{i, 1}; % e.g. saved
                    src_var = all_vars{i, 2}; % e.g. AthenaDelayComp_hit_history

                    % Extract the destination field from the source variable
                    split_name = strsplit(src_var, '_'); 
                    dest_field = strjoin(split_name(2:end), '_'); % eg. hit_hisotry

                    % initialize strcut with fields & correct values and
                    % then iteratively add each
                    switch src_struct
                        case 's'
                            data_source = saved;
                            if ~isfield(ratdata, dest_field)
                                ratdata.(dest_field) = [];
                            end

                        case 'sh'
                            data_source = saved_history;
                            if ~isfield(ratdata, dest_field)
                                ratdata.(dest_field) = {};
                            end
                        otherwise
                            error('Unknown data source.');
                    end

                    % Populate the data
                    % eg. ratdata.hit_history(end+1:end+n_trials) =
                    % saved.AthenaDelayComp_hit_history(1:n_trials)
                     ratdata.(dest_field)(end+1:end+n_trials) = data_source.(src_var)(1:n_trials);
                end
              
                % Write to CSV (gets overwritten & extended each session)
                saved_file_path = 'X:\jbreda\PWM_data_scrape\';
                file_name = [saved_file_path ratname '_trials_data.csv'];
                T = struct2table(structfun(@transpose,ratdata,'UniformOutput',false));
                writetable(T, file_name);

            end
        end
    end
    
end


%% HELPER FUNCTIONS

function stage = getTrainingStage(reward_type, stimuli_on, n_trials)
    % getTrainingStage: Determines the training stage for each trial
    % @param: reward_type - type of reward given
    % @param: stimuli_on - whether the stimuli were on
    % @param: n_trials - number of done trials for the session
    % @return: stage - stage of training

    % Check if the input arrays are empty
    if isempty(reward_type) || isempty(stimuli_on)
        warning('Input arrays should not be empty.');
        return;
    end
    
    % Check if n_trials is defined
    if isempty(n_trials)
        warning('Number of trials is not defined.');
        return;
    end
    
    % allocate & reformat
    stage = zeros(1, n_trials);
    stimuli_on = cell2mat(stimuli_on);
    
    % TODO- one psych pairs are found, add these to create a 5th stage

    for itrial = 1:n_trials
        if ~stimuli_on(itrial)
            stage(itrial) = 1;
        elseif strcmp(reward_type{itrial}, 'Always')
            stage(itrial) = 2;
        elseif strcmp(reward_type{itrial}, 'DelayedReward')
            stage(itrial) = 3;
        elseif strcmp(reward_type{itrial}, 'NoReward')
            stage(itrial) = 4;
        else
            stage(itrial) = nan;
        end
    end
end



function sound_dB = getSoundDb(sound_sigma, n_trials)

    % getSoundDb: Convert from sigma wave param value to dB
    %   this is done using inference performed by Chuck between
    %   the publically available rat dataset in dB and the brody
    %   lab acces to the raw sigma values
    %
    % @param: sound_sigma - sigma value for a1 or a1 (first, second sound
    % or sa, sb)
    % @param: n_trials - number of done trials for the session
    % 
    % @return: sound_dB - dB value for a1 (first sound or sa)
    % @return: a2_dB - dB value for a2 (second sound or sb)
    sound_sigma = cell2mat(sound_sigma(1:n_trials));
    sound_dB = round((8.0544.*log(sound_sigma)) + 99.9645);

end

    



