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
function [obsData, ephData] = doFrameDecoding_L5(obsData, trackData, allSettings)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function performs frame decoding for GPS L5 and Galileo E5a signals
%
% This is a TEMPORARY WRAPPER that uses the standard frame decoding
%
% TODO: Implement L5/E5a-specific frame decoding:
%   - GPS L5 CNAV message decoding
%   - Galileo E5a I/NAV message decoding
%   - Secondary code handling
%
% Inputs:
%   obsData         - Observation data
%   trackData       - Tracking results
%   allSettings     - Receiver settings
%
% Outputs:
%   obsData         - Updated observation data
%   ephData         - Ephemeris data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('WARNING: Using standard frame decoding as temporary solution\n');
fprintf('L5/E5a-specific frame decoding not yet implemented\n\n');

% Call the standard frame decoding function
try
    [obsData, ephData] = doFrameDecoding(obsData, trackData, allSettings);
catch ME
    fprintf('ERROR: Standard frame decoding failed\n');
    fprintf('Error message: %s\n', ME.message);
    fprintf('Returning empty ephemeris data\n');
    ephData = [];
end
