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
function settings = getSystemParameters_L5(settings)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions sets GPS L5 and Galileo E5a parameters and constants
% in the settings structure
%
%  Inputs:
%       settings - Receiver settings
%
%  Outputs:
%       settings - Updated receiver settings
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GPS L5 hardcoded parameters
% Reference: IS-GPS-705 Interface Specification
settings.gpsl5.codeLengthInChips = 10230;  % L5 code length
settings.gpsl5.codeFreqBasis = 10.23e6;    % 10.23 MHz code rate
settings.gpsl5.carrierFreq = 1176.45e6;    % L5 carrier frequency
settings.gpsl5.numberOfChannels = 12;
settings.gpsl5.preamble = [1 -1 -1 -1 1 -1 1 1];  % Same as L1
settings.gpsl5.bitDuration = 1;            % 100 bps data rate (1 ms/bit for 1kHz code rate)
settings.gpsl5.secondaryCode = [1 1 -1 1 -1 1 -1 -1 -1 -1];  % Neuman-Hoffman code for L5I (10 bits)
settings.gpsl5.preambleCorrThr = 153;
settings.gpsl5.preambleIntervall = 6000;   % May need adjustment for L5
settings.gpsl5.frameLength = 30000;
settings.gpsl5.frequencyStep = 0;
settings.gpsl5.modulationFactor = 1;
settings.gpsl5.bitSyncConfidenceLevel = 6;

% Galileo E5a hardcoded parameters
% Reference: Galileo OS SIS ICD
settings.gale5a.codeLengthInChips = 10230;  % E5a primary code length
settings.gale5a.codeFreqBasis = 10.23e6;    % 10.23 MHz code rate
settings.gale5a.carrierFreq = 1176.45e6;    % E5a carrier frequency (same as GPS L5)
settings.gale5a.numberOfChannels = 12;
settings.gale5a.preamble = [1 -1 1 -1 -1 1 -1 -1 1 1];  % E5a sync pattern
settings.gale5a.bitDuration = 1;            % Symbol rate (may need adjustment)
settings.gale5a.secondaryCode = [1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 1 1 1 1 1 1];  % E5a secondary code (20 bits)
settings.gale5a.preambleCorrThr = 9.99;
settings.gale5a.preambleIntervall = 250;
settings.gale5a.frameLength = 10000;
settings.gale5a.frequencyStep = 0;
settings.gale5a.modulationFactor = 1;
settings.gale5a.bitSyncConfidenceLevel = 6;

% Physical constants (same as original)
settings.const.PI = 3.1415926535898;
settings.const.SPEED_OF_LIGHT = 2.99792458e8;

settings.const.SECONDS_IN_MINUTE = 60;
settings.const.SECONDS_IN_HOUR = 3600;
settings.const.SECONDS_IN_DAY = 86400;
settings.const.SECONDS_IN_HALF_WEEK = 302400;
settings.const.SECONDS_IN_WEEK = 604800;
settings.const.EARTH_SEMIMAJORAXIS = 6378137;
settings.const.EARTH_FLATTENING = 1/298.257223563;
settings.const.EARTH_GRAVCONSTANT = 3.986005e14;
settings.const.EARTH_WGS84_ROT = 7.2921151467E-5;
settings.const.C20 = -1082.62575e-6; % 2nd zonal harmonic of ellipsoid
settings.const.A_REF = 26559710;                        % CNAV2 Reference semi-major axis (meters)
settings.const.OMEGA_REFDOT = -2.6e-9*3.1415926535898;  % CNAV2 Reference rate of right ascension

settings.sys.nrOfChannels = 0;

% Total number of signals enabled
settings.sys.nrOfSignals = length(settings.sys.enabledSignals);

% Let's loop over all enabled signals
for i = 1:settings.sys.nrOfSignals

    % Extract block of parameters for one signal from settings
    signal = settings.sys.enabledSignals{i};
    paramBlock = settings.(signal);

    % Set additional block parameters
    paramBlock.samplesPerChip     = paramBlock.samplingFreq/paramBlock.codeFreqBasis; % Samples per chip
    paramBlock.samplesPerCode = round(paramBlock.samplingFreq / (paramBlock.codeFreqBasis / paramBlock.codeLengthInChips)); % Samples per each code epoch
    paramBlock.bytesPerSample = (paramBlock.sampleSize)/8;

    paramBlock.codeLengthMs = 1000 * paramBlock.codeLengthInChips / paramBlock.codeFreqBasis;

    paramBlock.intermediateFreq = paramBlock.carrierFreq - paramBlock.centerFrequency;
    paramBlock.signal = signal;

    % If user requests us to read 0 ms it means we need to read the whole file
    if(settings.sys.msToProcess == 0)
        f=dir(paramBlock.rfFileName);
        if ~isempty(f)
            fileBytes = f.bytes;
            samplesPerMs = paramBlock.samplingFreq/1000; % Number of samples per millisecond
            % Let's calculate how much data to skip and to read from the file
            msInFile = fileBytes/(paramBlock.bytesPerSample*samplesPerMs);
            maxMsToRead = msInFile - settings.sys.msToSkip;
            settings.sys.msToProcess = floor(maxMsToRead);
        else
            error('RF data file not found: %s', paramBlock.rfFileName);
        end
    end
    paramBlock.numberOfBytesToSkip = paramBlock.bytesPerSample*paramBlock.samplingFreq/1000*settings.sys.msToSkip;
    paramBlock.numberOfBytesToRead = paramBlock.bytesPerSample*paramBlock.samplingFreq/1000*settings.sys.msToProcess;

    if(paramBlock.complexData == true)
        paramBlock.dataType = strcat('int',num2str(paramBlock.sampleSize/2));
    else
        paramBlock.dataType = strcat('int',num2str(paramBlock.sampleSize));
    end

    % Add number of channels for one signal to total number of channels
    settings.sys.nrOfChannels = settings.sys.nrOfChannels + paramBlock.numberOfChannels;

    % Copy block of parameters back to settings
    settings.(signal) = paramBlock;

    disp(strcat(signal,' Enabled'));
end
