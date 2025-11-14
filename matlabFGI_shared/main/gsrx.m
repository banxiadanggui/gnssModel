%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copyright 2015-2021 Finnish Geospatial Research Institute FGI, National
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
function [] = gsrx(varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main function for the FGI-GSRx matlab software receiver
%
% Input (optional):
%   varargin   -   Name of user parameter file
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
    'Welcome to:  FGI-GSRx software GNSS receiver\n', ...
    'by the Finnish Geospatial Research Institute\n\n']);
fprintf('                   -------------------------------\n\n');

if isempty(varargin)
    % Initialize receiver settings (parameters)
    settings = readSettings(varargin);
else
    settings = readSettings(varargin{1});
end
% Read existing pre processed data file
if(settings.sys.loadDataFile == true)
    newSettings = settings; % Copy parameters to temporary variable
    load(settings.sys.dataFileIn);
    settings = newSettings; % Overwrite parameters in data file
end

% Generate spectrum plots
if settings.sys.plotSpectra == 1
    generateSpectra(settings);
end

% Execute acquisition if results not already available
if(~exist('acqData', 'var'))
    % Check if we can skip acquisition by loading from existing tracking files
    if settings.sys.parallelChannelTracking
        trackDataFilePath = settings.sys.trackDataFilePath;
        % Ensure it's a character vector, not string array
        if iscell(trackDataFilePath)
            trackDataFilePath = trackDataFilePath{1};
        end
        trackDataFilePath = char(trackDataFilePath);

        existingTrackFiles = dir([trackDataFilePath,'trackData_*.mat']);
        if ~isempty(existingTrackFiles)
            fprintf('\n=== Skipping acquisition - loading from existing tracking files ===\n');
            firstTrackFile = fullfile(trackDataFilePath, existingTrackFiles(1).name);
            fprintf('Loading acqData from: %s\n', firstTrackFile);
            load(firstTrackFile, 'acqData');
            fprintf('acqData loaded successfully.\n');
        else
            acqData = doAcquisition(settings);
        end
    else
        acqData = doAcquisition(settings);
    end
end

% Plot acquisition results
if settings.sys.plotAcquisition == 1
   % Loop over all signals
    for i = 1:settings.sys.nrOfSignals
        signal = settings.sys.enabledSignals{i};
        plotAcquisition(acqData.(signal),settings, char(signal));
    end
end

% Execute tracking if results not allready available
if(~exist('trackData', 'var'))
    tic;
    if (settings.sys.parallelChannelTracking)
        % Check if parallel tracking files already exist
        fprintf('\n=== Checking for existing parallel tracking results ===\n');
        trackDataFilePath = settings.sys.trackDataFilePath;
        % Ensure it's a character vector, not string array
        if iscell(trackDataFilePath)
            trackDataFilePath = trackDataFilePath{1};
        end
        trackDataFilePath = char(trackDataFilePath);
        fprintf('Track data path: %s\n', trackDataFilePath);

        firstSignal = settings.sys.enabledSignals{1};
        % Ensure firstSignal is a character vector
        if iscell(firstSignal)
            firstSignal = firstSignal{1};
        end
        firstSignal = char(firstSignal);

        firstSatId = 0;
        for channelNr = 1:length(acqData.(firstSignal).channel)
            if acqData.(firstSignal).channel(channelNr).bFound == 1
                firstSatId = acqData.(firstSignal).channel(channelNr).SvId.satId;
                break;
            end
        end
        fprintf('First signal: %s, First satellite ID: %d\n', firstSignal, firstSatId);

        % Construct path to first tracking file (no strjoin needed since all are char)
        firstTrackFile = [trackDataFilePath,'trackData_',firstSignal,'_Satellite_ID_',num2str(firstSatId),'.mat'];
        fprintf('Checking for file: %s\n', firstTrackFile);

        if (~exist(firstTrackFile, 'file'))
            fprintf('File NOT found - executing Step 1 (split and generate batch)\n');
            % Step 1: Initialize and generate batch file
            trackDataFileName = initializeAndSplitTrackingPerChannel(acqData, settings);
            doTrackingParallel(trackDataFileName,settings);
            fprintf('\n=== Step 1 Complete ===\n');
            fprintf('Please run the batch file: %s\n', settings.sys.batchFileNameToRunParallelTracking);
            fprintf('After batch completes, run gsrx again to merge results.\n');
            return;
        else
            fprintf('File FOUND - executing Step 3 (merge results)\n');
            % Step 3: Merge parallel tracking results
            fprintf('\n=== Step 3: Merging parallel tracking results ===\n');
            trackData = combineSingleTrackChannelData(settings);
        end
    else
        trackData = doTracking(acqData, settings);
    end
    trackData.trackingRunTime = toc;
end

% Plot tracking results
if settings.sys.plotTracking == 1                
    plotTracking(trackData, settings);    
end

% Convert track data to useful observations for navigation if data not already available
if(~exist('obsData', 'var'))
    obsData = generateObservations(trackData, settings);
end

% Execute frame decoding. Needed for time stamps at least
if(~exist('ephData', 'var'))
    [obsData, ephData] = doFrameDecoding(obsData, trackData, settings);
end

% Execute navigation
if(~exist('navData', 'var'))
    [obsData,satData,navData] = doNavigation(obsData, settings, ephData);
end

% Calculate and output statistics
% True values
trueLat = settings.nav.trueLat; 
trueLong = settings.nav.trueLong;
trueHeight = settings.nav.trueHeight;

% Calculate statistics
if(~exist('statResults', 'var'))
    statResults = calcStatistics(navData,[trueLat trueLong trueHeight],settings.nav.navSolPeriod,settings.const);
end

% Save results so far to file
if(settings.sys.saveDataFile == true)
    save(settings.sys.dataFileOut,'settings','acqData','ephData','trackData','obsData','satData','navData','statResults');
end

% Output statistics
statResults.hor
statResults.ver
statResults.dop
statResults.RMS3D
