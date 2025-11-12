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
function [] = gsrx_L5(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function for GPS L5 / Galileo E5a signal processing
%
% This is a specialized version of FGI-GSRx for L5/E5a signals
%
% Input (optional):
%   varargin   -   Name of user parameter file
%
% Usage:
%   gsrx_L5('..\param\test_param_FGISpoofRepo_GPSL5_GalE5a_UTD.txt')
%
% Dependencies:
%   - readSettings_L5.m
%   - getSystemParameters_L5.m
%   - generateGPSL5Code.m (for PRN code generation)
%   - generateGalileoE5aCode.m (for PRN code generation)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Setting some display font settings.
set(0,'defaultaxesfontsize',10);
set(0,'defaultlinelinewidth',2);

% Clean up the environment
close all;
clc;
clearvars -except varargin

% Set number format
format ('compact');
format ('long', 'g');

% Print startup text
fprintf(['\n',...
    '=====================================================\n',...
    '  FGI-GSRx L5/E5a Signal Processing Module\n', ...
    '  GPS L5 & Galileo E5a Receiver\n', ...
    '  Finnish Geospatial Research Institute\n',...
    '=====================================================\n\n']);

if isempty(varargin)
    % Initialize receiver settings (parameters)
    settings = readSettings_L5(varargin);
else
    settings = readSettings_L5(varargin{1});
end

% Read existing pre processed data file
if(settings.sys.loadDataFile == true)
    newSettings = settings; % Copy parameters to temporary variable
    load(settings.sys.dataFileIn);
    settings = newSettings; % Overwrite parameters in data file
end

% Generate spectrum plots
if settings.sys.plotSpectra == 1
    fprintf('Generating spectrum plots...\n');
    generateSpectra(settings);
end

% Define ephData if not available
if(~exist('ephData', 'var'))
    ephData = [];
end

% Execute acquisition if results not already available
if(~exist('acqData', 'var'))
    fprintf('Starting L5/E5a signal acquisition...\n');
    acqData = doAcquisition_L5(settings);
end

% Plot acquisition results
if settings.sys.plotAcquisition == 1
   % Loop over all signals
    for i = 1:settings.sys.nrOfSignals
        signal = settings.sys.enabledSignals{i};
        plotAcquisition(acqData.(signal),settings, char(signal));
    end
end

% Save available results so far to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData');
end

% Execute tracking if results not already available
if(~exist('trackData', 'var'))
    fprintf('Starting L5/E5a signal tracking...\n');
    tic;
    if (settings.sys.parallelChannelTracking)
        if (~exist('trackResults', 'var'))
            trackDataFileName = initializeAndSplitTrackingPerChannel(acqData, settings);
            doTrackingParallel(trackDataFileName,settings);
            return;
        else
            trackData = combineSingleTrackChannelData(settings);
        end
    else
        trackData = doTracking_L5(acqData, settings);
    end
    trackData.trackingRunTime = toc;
    fprintf('Tracking completed in %.2f seconds\n', trackData.trackingRunTime);
end

% Save results so far to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData','trackData');
end

% Plot tracking results
if settings.sys.plotTracking == 1
    plotTracking(trackData, settings);
end

% Convert track data to useful observations for navigation if data not already available
if(~exist('obsData', 'var'))
    fprintf('Generating observations...\n');
    obsData = generateObservations(trackData, settings);
end

% Save results so far to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData','trackData','obsData');
end

% Execute frame decoding. Needed for time stamps at least
fprintf('Decoding L5/E5a navigation frames...\n');
[obsData, ephData] = doFrameDecoding_L5(obsData, trackData, settings);

% Save results so far to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData','trackData','obsData');
end

% Execute navigation
fprintf('Computing navigation solution...\n');
[obsData,satData,navData] = doNavigation(obsData, settings, ephData);

% Save final results to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData','trackData','obsData','satData','navData');
end

% Calculate and output statistics
% True values
trueLat = settings.nav.trueLat;
trueLong = settings.nav.trueLong;
trueHeight = settings.nav.trueHeight;

% Calculate statistics
statResults = calcStatistics(navData,[trueLat trueLong trueHeight],settings.nav.navSolPeriod,settings.const);

% Output statistics
fprintf('\n=====================================================\n');
fprintf('  L5/E5a Navigation Results\n');
fprintf('=====================================================\n');
fprintf('Horizontal Statistics:\n');
disp(statResults.hor);
fprintf('\nVertical Statistics:\n');
disp(statResults.ver);
fprintf('\nDOP Values:\n');
disp(statResults.dop);
fprintf('\n3D RMS Error: %.2f m\n', statResults.RMS3D);
fprintf('=====================================================\n\n');
