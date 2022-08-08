%% UPDATE_WATER_INFO 
% a function that allows you to open the settings file for an animal, see
% what their most recent water volumes are, update them if you want and 
% save a new settings file
% 
% Inputs:
%
%   animalids  : (optional) cell array of strings, e.g. {'R500', 'R501'}.
%                Defaults to list from get_animal_ids()
% Outputs:
%
%   if water is updated, a new settings file will be saved for that animal
%   on the next date
%
% Example usage:
%
%

function message = update_water_info(animalids)
    
    load_solo_helper_fxs(); % load functions
    if ~exist('animalids', 'var'); animalids = get_animal_ids(); end
    
    for ianimal = 1 : length(animalids)
        % get the latest settings file
        filepaths = dir(get_animal_settings_dir(animalids{ianimal}));
        filepath = [filepaths(end).folder '/' filepaths(end).name];
        recdate = filepath(end - 10 : end - 4);
    end % STOPPED HERE
    message = 'done';
        
end

