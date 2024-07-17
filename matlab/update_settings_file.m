%% UPDATE_SETTLING_IN_DUR
% a function that allows you to open the settings file for an animal, see
% what their most recent settling in dur is, and update update it if you 
% want and save a new settings file
% 
% Inputs:
%
%   animalid  : animal id as a string
%
% Outputs:
%
%   if settling in dur is updated, a new settings file will be saved for 
%   that animal on the next date
%
% Example usage:
% update_settling_in_dur('R500')
%

function message = update_settings_file(animal_id)
    
    load_solo_helper_fxs(); 
    
    % get the latest settings file for the animal. Have to do end-1 since
    % the last sorted name is '.' 
    file_paths = dir(get_animal_settings_dir(animal_id));
    [~, index] = sort([file_paths.datenum]); % Sort by the 'datenum' field
    file_paths_by_date = file_paths(index);

    if file_paths_by_date(end).name == '.'
        fname = file_paths_by_date(end-1).name;
    else
        fname = file_paths_by_date(end).name;
    end
    
    file_path = [file_paths_by_date(end).folder '/' fname];
    rec_date = file_path(end - 10 : end - 4);
    % load data and get current settling in dur
    data = load(file_path);

    fprintf('\nAnimal: %s, %s\n', animal_id, rec_date);
    dbstop in update_settings_file at 44
    
    %% optional additional water update
    reply = input('\nDo you want to change the water volumes as well? y/n : ', "s");
    
    if reply == "y"
        
        left_water = data.saved.WaterValvesSection_Left_volume;
        right_water = data.saved.WaterValvesSection_Right_volume;

        % print current volumes
        fprintf('\nleft volume in uL = %.2f\n', left_water);
        fprintf('right volume in uL = %.2f\n\n', right_water);
        
        % update volumes based on input
        updated_left_volume = input('\nleft volume in uL: ');
        data.saved.WaterValvesSection_Left_volume = updated_left_volume;
        updated_right_volume = input('right volume in uL: ');
        data.saved.WaterValvesSection_Right_volume = updated_right_volume;
        
        message = sprintf('water: L %.1f -> %.1f, R %.1f -> %.1f',...
            left_water, updated_left_volume, right_water, updated_right_volume);         
    else 
        message = sprintf('other updates made!');    
    end
   
    % generate new file name
    updated_file_path = generate_modified_filepath(file_path);
    fprintf('\nOld file: %s\n', file_path(end - 36 : end - 4));
    fprintf('New file: %s\n', updated_file_path(end - 36 : end - 4));
    input('\nPress enter to confirm (or ctrl-c to cancel)');
    save(updated_file_path, '-struct', 'data');
      
    
end

