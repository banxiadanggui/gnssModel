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
function trackDataCombined = combineSingleTrackChannelData(allSettings)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function combines individual satellite tracking results into a
% unified tracking data structure.
%
% Inputs:
%   allSettings     - Receiver settings
%
% Outputs:
%   trackDataCombined - Combined tracking results for all signals
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trackDataFilePath = allSettings.sys.trackDataFilePath;
% Ensure it's a character vector, not string array
if iscell(trackDataFilePath)
    trackDataFilePath = trackDataFilePath{1};
end
trackDataFilePath = char(trackDataFilePath);

fprintf('Combining tracking results from: %s\n', trackDataFilePath);

% Load acqData from first tracking file
acqDataLoaded = false;

for signalNr = 1:allSettings.sys.nrOfSignals % Loop over all signals
    signal = allSettings.sys.enabledSignals{signalNr};
    % Ensure signal is a character vector
    if iscell(signal)
        signal = signal{1};
    end
    signal = char(signal);

    fprintf('\nProcessing signal: %s\n', signal);

    % Load acqData if not already loaded
    if ~acqDataLoaded
        trackFiles = dir([trackDataFilePath,'trackData_',signal,'_Satellite_ID_*.mat']);
        if isempty(trackFiles)
            error('No tracking files found for signal %s in %s', signal, trackDataFilePath);
        end
        firstTrackFile = fullfile(trackDataFilePath, trackFiles(1).name);
        fprintf('Loading acqData from: %s\n', firstTrackFile);
        load(firstTrackFile, 'acqData');
        acqDataLoaded = true;
    end

    % Collect all channel data for this signal
    channelArray = [];
    validChannelCount = 0;

    for channelNr = 1:length(acqData.(signal).channel)
        if acqData.(signal).channel(channelNr).bFound == 1
            satId = acqData.(signal).channel(channelNr).SvId.satId;
            trackDataFileName = [trackDataFilePath,'trackData_',signal,'_Satellite_ID_',num2str(satId),'.mat'];

            if ~exist(trackDataFileName, 'file')
                fprintf('  WARNING: File not found: %s\n', trackDataFileName);
                continue;
            end

            % Check what variables are in the file
            fileVars = whos('-file', trackDataFileName);
            varNames = {fileVars.name};

            % Load the appropriate variable
            if ismember('trackResults', varNames)
                tempData = load(trackDataFileName, 'trackResults');
                singleChannel = tempData.trackResults.(signal).channel;
                if validChannelCount == 0
                    % Use first file to initialize metadata
                    trackDataCombined.(signal) = tempData.trackResults.(signal);
                end
            elseif ismember('trackResultsSingle', varNames)
                fprintf('  WARNING: Loading unprocessed data from: %s\n', trackDataFileName);
                tempData = load(trackDataFileName, 'trackResultsSingle');
                singleChannel = tempData.trackResultsSingle.(signal).channel;
                if validChannelCount == 0
                    % Use first file to initialize metadata
                    trackDataCombined.(signal) = tempData.trackResultsSingle.(signal);
                end
            else
                fprintf('  WARNING: No valid tracking data in: %s\n', trackDataFileName);
                continue;
            end

            validChannelCount = validChannelCount + 1;

            % Build channel array
            if validChannelCount == 1
                channelArray = singleChannel;
            else
                channelArray(validChannelCount) = singleChannel;
            end

            fprintf('  Loaded channel %d (Sat ID: %d)\n', validChannelCount, satId);
        end
    end

    % Assign the complete channel array
    if validChannelCount > 0
        trackDataCombined.(signal).channel = channelArray;
        trackDataCombined.(signal).nrObs = validChannelCount;
        fprintf('Successfully combined %d channels for %s\n', validChannelCount, signal);
    else
        error('No valid channels found for signal %s', signal);
    end
end

fprintf('\nCombining complete!\n');
