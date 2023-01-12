%% UPDATE_REWARD_TYPE
% a function that allows you to open the settings file for an animal, see
% what their most recent reward_type is, and update update it if you want and 
% save a new settings file
% 
% Inputs:
%
%   animalid  : animal id as a string
%
% Outputs:
%
%   if reward_type is updated, a new settings file will be saved for that 
%   animal on the next date
%
% Example usage:
% update_reward_type('R500')
%

function message = update_reward_type(animal_id)
    
    load_solo_helper_fxs(); 
    
    % get the latest settings file for the animal
    file_paths = dir(get_animal_settings_dir(animal_id));
    file_path = [file_paths(end).folder '/' file_paths(end).name];
    rec_date = file_path(end - 10 : end - 4);
    
    % load data and get current water volume
    data = load(file_path);
    reward_type = data.saved.ShapingSection_reward_type;
    
    fprintf('\nAnimal: %s, %s\n', animal_id, rec_date);
    fprintf('\ncurrent reward type is: ''%s''\n', reward_type);
    
    reply = input('\nDo you want to change the reward type? y/n : ', "s");
    
    if reply ~= "y"
        return
    end
        
    % update reward type based on input
    updated_reward_type = input('\nupdated reward type, options are ''poke'' or ''give'': '); 
  
    if ~(strcmp(updated_reward_type,'poke')) && ~(strcmp(updated_reward_type, 'give'))
        % ensure correct options and format (e.g. give needs to be 'give')
        error('reward type not valid! need to be either ''give'' or ''poke'' in str quotes');
    end
    
    data.saved.ShapingSection_reward_type = updated_reward_type;
    
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
        
        message = sprintf('water: L %.1f -> %.1f, R %.1f -> %.1f \nreward type: %s -> %s',...
            left_water, updated_left_volume, right_water, updated_right_volume,...
            reward_type, updated_reward_type);         
    else 
        message = sprintf('reward type: %s -> %s',...
        reward_type, updated_reward_type);     
    end
   
    % generate new file name
    updated_file_path = generate_modified_filepath(file_path);
    fprintf('\nOld file: %s\n', file_path(end - 35 : end - 4));
    fprintf('New file: %s\n', updated_file_path(end - 35 : end - 4));
    input('\nPress enter to confirm (or ctrl-c to cancel)');
%     save(updated_file_path, '-struct', 'data');
      
    
end

