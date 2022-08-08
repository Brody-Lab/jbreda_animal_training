%% LOAD_SOLO_HELPER_FUNCTIONS
%
% use this function to load all the other helper funtions in this notebook
% that can be used when accessing solo data/settings in the brody lab
% cup drive
%
% note : many of these functions have been taken from the
% jyanar_training_scripts repository and modified to work with my animals
%
% written by JRB August 2022



function message = load_solo_helper_fxs

    assignin('caller','get_animal_ids',@get_animal_ids);
    assignin('caller','get_experimenter',@get_experimenter);
    assignin('caller','get_animal_settings_dir',@get_animal_settings_dir);
    
    message = 'Solo helper functions imported to workspace';
    
end

%% GET_ANIMAL_IDS
% function to grab animals currently being tranied on DMS
% protocol given different criteria
%
% inputs:
% -------  
%   criteria : (optional) if empty, will grab all animals.
%               as of 2022-08 the 'rats' criteria is just to make
%               things run, will be updated when needed
%
% returns:
% --------
%   animalids : a cell array of strings with animal ids
%

function animalids = get_animal_ids(criteria)

    animalids = {'R500', 'R501', 'R502',...
                 'R503', 'R600'};
    
    if ~exist('criteria', 'var'); criteria = 'all'; end
    switch criteria
        case 'rats'
            animalids = {'R001', 'R002', 'R003', 'R004'};
    end
    
    % TODO- expand with criteria (e.g. cohort 1, rats etc if 
    % helpful
    
end

%% GET_EXPERIMENTER
% Given the first intial of the animalid, this function will return the 
% correct experimenter name used in the file path
%
% inputs:
% -------
%   animalprefix : first character of an animal id e.g. 'R'
%
% returns:
% --------
%   experimenter : name of exerimenter associated with animal as a string
% 
function experimenter = get_experimenter(animalprefix)
    if animalprefix == 'R'
        experimenter = 'JessB';
    elseif animalprefix == 'I'
        experimenter = 'Josh';
    elseif (animalprefix == 'D') || (animalprefix == 'E')
        experimenter = 'Emily';
    elseif animalprefix == 'Y'
        experimenter = 'Jorge';
    else
        error('animalprefix must be for a valid animal');
    end
end

%% GET_ANIMAL_SETTINGS_DIR

% Function to find & retturn the directory on Brody Lab cup where animals
% settings file is located
% 
% inputs:
% -------  
%   animalid : string of animal name
%
% returns:
% --------
%   dirpath : directory path of settings for animalid as a string

function dirpath = get_animal_settings_dir(animalid)
    % get experimenter
    experimenter = get_experimenter(animalid(1));
    % create path
    dirpath = ['X:/RATTER/SoloData/Settings/' experimenter '/' animalid '/'];
end




















