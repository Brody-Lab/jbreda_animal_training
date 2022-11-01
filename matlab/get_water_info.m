%% get_water_info
%
% written by JRB 07-28-2022
%
% Script to get todays (or the most recent) water information for
% a list of mutable animals. Checks the amount consumed & target
% restriction to ensure protocol guidlines are being followed
%
% TODO: turn into a function, make flexible to a specified number
%       of dates, write automatically to a log

animals = {'R500', 'R501', 'R502', 'R503', 'R600'};
% animals = {'R501', 'R502', 'R600'};

for ianimal = 1 : length(animals)
   
    [w_date, pub_volume, target] = bdata(['select date, volume, percent_target from ',...
        'ratinfo.water where rat="', animals{ianimal}, '" order by date']);
    [rig_volume] = bdata(['select totalvol from ',...
        'ratinfo.rigwater where ratname="', animals{ianimal}, '" order by dateval']);
    [m_date, mass] = bdata(['select date, mass from ',...
        'ratinfo.mass where ratname="', animals{ianimal}, '" order by date']);
    
    % sanity check that mass & water entries are from the same day
     if ~ismember(w_date(end), m_date(end))
%          disp(animals{ianimal})
%          disp('water date:')
%          disp(w_date(end))
%          disp('mass date:')
%          disp(m_date(end))

%          sprintf('water date: %s, mass_date :%s', water_date, mass_date);
%          error('mass & water date do not match!')
     end
    
     % calclulate totals
     water_drunk = rig_volume(end) + pub_volume(end);
     target_volume = mass(end) * target(end)/100; % target set in rat registry comments waterXpub
     if water_drunk >= target_volume, complete='TRUE';
     else, complete='!!!FALSE!!!!'; end
        
    fprintf(['Animal:[[%s]], rig: %.3f mL, pub : %.3f mL, restrict: %.2f  \n',...
            '            total: %0.3f mL, target: %0.3f mL, complete: %s \n'],...
            animals{ianimal}, rig_volume(end), pub_volume(end), target(end),...
            water_drunk, target_volume, complete);
end
