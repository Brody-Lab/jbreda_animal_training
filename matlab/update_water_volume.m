%% UPDATE_WATER_VOLUME
% a function that allows you to open the settings file for an animal, see
% what their most recent water volumes are, update them if you want and 
% save a new settings file
% 
% Inputs:
%
%   animalid  : animal id as a tring
%
% Outputs:
%
%   if water is updated, a new settings file will be saved for that animal
%   on the next date
%
% Example usage:
% update_water_volume('R500')
%

function message = update_water_volume(animal_id)
    
    load_solo_helper_fxs(); 
    
    % get the latest settings file for the animal
    file_paths = dir(get_animal_settings_dir(animal_id));
    file_path = [file_paths(end).folder '/' file_paths(end).name];
    rec_date = file_path(end - 10 : end - 4);
    
    % load data and get current water volume
    data = load(file_path);
    left_water = data.saved.WaterValvesSection_Left_volume;
    right_water = data.saved.WaterValvesSection_Right_volume;
    
    fprintf('\nAnimal: %s, %s\n', animal_id, rec_date);
    fprintf('\nleft volume in uL = %.2f\n', left_water);
    fprintf('right volume in uL = %.2f\n\n', right_water);
    
    reply = input('Do you want to change the water volumes? y/n : ', "s");
    
    if reply == "n"
        return
    end
        
    % update volumes based on input
    updated_left_volume = input('\nleft volume in uL: ');
    data.saved.WaterValvesSection_Left_volume = updated_left_volume;
    updated_right_volume = input('right volume in uL: ');
    data.saved.WaterValvesSection_Right_volume = updated_right_volume;
        
    % generate new file name
    updated_file_path = generate_modified_filepath(file_path);
    fprintf('\nOld file: %s\n', file_path(end - 35 : end - 4));
    fprintf('New file:   %s\n', updated_file_path(end - 35 : end - 4));
    input('\nPress enter to confirm (or ctrl-c to cancel)');
    save(updated_file_path, '-struct', 'data');
      
    message = sprintf('water: L %.1f -> %.1f, R %.1f -> %.1f',...
        left_water, updated_left_volume, right_water, updated_right_volume);     
end

