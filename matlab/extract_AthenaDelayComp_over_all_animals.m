% List of rat names as strings
rat_names = {'W051', 'W060', 'W061'};%,...
%             'W062', 'W065', 'W066',...
%             'W068', 'W072', 'W073',...
%             'W074', 'W075', 'W078',...
%             'W080', 'W081', 'W082',...
%             'W083', 'W088', 'W089',...
%             'W094'};

% Iterate over each rat name, calling the function and storing the result
for i = 1:length(rat_names)
    extract_AthenaDelayComp_data(rat_names{i});
end


% issues