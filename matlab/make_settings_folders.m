%% Make settings folders

% Define the parent directory and folder names
parentDirectory = 'X:\RATTER\SoloData\Settings\JessB';
startNumber = 30; % Start number
stopNumber = 39;  % Stop number

% Create the folders
for i = startNumber:stopNumber
    folderName = sprintf('R%03d', i);
    folderPath = fullfile(parentDirectory, folderName);
    
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
        fprintf('Created folder: %s\n', folderPath);
    else
        fprintf('Folder already exists: %s\n', folderPath);
    end
end