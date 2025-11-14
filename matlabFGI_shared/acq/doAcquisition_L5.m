%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright 2015-2025 Finnish Geospatial Research Institute FGI, National
%% Land Survey of Finland. This file is part of FGI-GSRx software-defined
%% receiver. FGI-GSRx is a free software: you can redistribute it and/or
%% modify it under the terms of the GNU General Public License as published
%% by the Free Software Foundation, either version 3 of the License, or any
%% later version. FGI-GSRx software receiver is distributed in the hope
%% that it will be useful, but WITHOUT ANY WARRANTY, without even the
%% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%% See the GNU General Public License for more details. You should have
%% received a copy of the GNU General Public License along with FGI-GSRx
%% software-defined receiver. If not, please visit the following website
%% for further information: https://www.gnu.org/licenses/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function acqResults = doAcquisition_L5(allSettings)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function performs signal acquisition for GPS L5 and Galileo E5a signals
%
% This is a TEMPORARY WRAPPER that uses the standard acquisition function
%
% TODO: Implement L5/E5a-specific acquisition with:
%   - 10230-chip code handling
%   - Longer coherent integration (10ms)
%   - More non-coherent integration rounds (10)
%   - Wider frequency search range
%
% Inputs:
%   allSettings     - receiver settings
%
% Outputs:
%   acqResults      - Results of signal acquisition
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('WARNING: Using standard acquisition function as temporary solution\n');
fprintf('L5/E5a-specific acquisition not yet implemented\n');
fprintf('This may not work correctly with L5/E5a signals\n\n');

% Call the standard acquisition function
% This will work if the signal processing chain is compatible
try
    acqResults = doAcquisition(allSettings);
catch ME
    fprintf('ERROR: Standard acquisition failed\n');
    fprintf('Error message: %s\n', ME.message);
    fprintf('\nTo fix this, you need to:\n');
    fprintf('1. Create proper L5/E5a acquisition implementation\n');
    fprintf('2. Or ensure PRN code generation functions are available\n');
    fprintf('3. Check that RF data file exists and is accessible\n');
    rethrow(ME);
end
