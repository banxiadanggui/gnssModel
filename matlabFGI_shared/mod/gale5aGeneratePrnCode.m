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
function codeReplica = gale5aGeneratePrnCode(PRN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates Galileo E5a satellite PRN code (10230 chips)
%
% Reference: Galileo OS SIS ICD
%
% Inputs:
%   PRN         - PRN number for which codes will be generated (1-36).
%
% Outputs:
%   codeReplica - Generated code replica in chips for the given PRN number
%                 Size: 1 x 10230
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Galileo E5a uses similar structure to GPS L5
% Code length: 10230 chips = 10.23 Mcps * 1 ms

% PRN-specific initial states for register B
% These are example values - should be replaced with actual Galileo values
% from Galileo OS SIS ICD Table 56-57
XBInitial = [
    bin2dec('1000010101100'); % PRN 1
    bin2dec('1010101110111'); % PRN 2
    bin2dec('1101000111001'); % PRN 3
    bin2dec('1111100001011'); % PRN 4
    bin2dec('1100110110101'); % PRN 5
    bin2dec('1011111010000'); % PRN 6
    bin2dec('1110001001100'); % PRN 7
    bin2dec('1010111011010'); % PRN 8
    bin2dec('1100101100011'); % PRN 9
    bin2dec('1001011110101'); % PRN 10
    bin2dec('1111010001110'); % PRN 11
    bin2dec('1011000101011'); % PRN 12
    bin2dec('1100111000110'); % PRN 13
    bin2dec('1110100011001'); % PRN 14
    bin2dec('1010001111101'); % PRN 15
    bin2dec('1101110010010'); % PRN 16
    bin2dec('1111001010110'); % PRN 17
    bin2dec('1100011101000'); % PRN 18
    bin2dec('1010110001110'); % PRN 19
    bin2dec('1110111100101'); % PRN 20
    bin2dec('1001110011100'); % PRN 21
    bin2dec('1011010111001'); % PRN 22
    bin2dec('1101001011110'); % PRN 23
    bin2dec('1111101100010'); % PRN 24
    bin2dec('1100100010111'); % PRN 25
    bin2dec('1010011110000'); % PRN 26
    bin2dec('1110110001100'); % PRN 27
    bin2dec('1001101100010'); % PRN 28
    bin2dec('1011110001110'); % PRN 29
    bin2dec('1101011010101'); % PRN 30
    bin2dec('1111110111001'); % PRN 31
    bin2dec('1100101101011'); % PRN 32
    bin2dec('1010111001110'); % PRN 33
    bin2dec('1110010110010'); % PRN 34
    bin2dec('1001111010111'); % PRN 35
    bin2dec('1011100110011'); % PRN 36
];

% Initialize output
codeReplica = zeros(1, 10230);

% Initialize XA register (all ones for E5a)
XA = ones(1, 13);

% Initialize XB register with PRN-specific initial state
if PRN <= length(XBInitial)
    XBInit = XBInitial(PRN);
else
    error('PRN must be between 1 and 36 for Galileo E5a');
end

XB = zeros(1, 13);
for i = 1:13
    XB(i) = bitget(XBInit, 14-i);
end

% Generate 10230 chips
% Using similar structure to GPS L5
for chip = 1:10230
    % Output is XOR of the last bits of XA and XB
    codeReplica(chip) = xor(XA(13), XB(13));

    % XA feedback: same as GPS L5
    % Polynomial: 1 + X^9 + X^10 + X^12 + X^13
    newXA = xor(xor(xor(XA(9), XA(10)), XA(12)), XA(13));

    % XB feedback: same as GPS L5
    % Polynomial: 1 + X + X^3 + X^4 + X^6 + X^7 + X^8 + X^12 + X^13
    newXB = xor(xor(xor(xor(xor(xor(xor(XB(1), XB(3)), XB(4)), XB(6)), XB(7)), XB(8)), XB(12)), XB(13));

    % Shift registers
    XA = [newXA XA(1:12)];
    XB = [newXB XB(1:12)];
end

% Convert from 0/1 to -1/+1
zeroIndices = find(codeReplica(:)==0);
codeReplica(zeroIndices) = -1;
