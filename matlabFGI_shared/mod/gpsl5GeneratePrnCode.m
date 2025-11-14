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
function codeReplica = gpsl5GeneratePrnCode(PRN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates GPS L5 satellite PRN code (10230 chips)
%
% Reference: IS-GPS-705 Interface Specification
%
% Inputs:
%   PRN         - PRN number for which codes will be generated (1-32).
%
% Outputs:
%   codeReplica - Generated code replica in chips for the given PRN number
%                 Size: 1 x 10230
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GPS L5 uses two 13-bit Linear Feedback Shift Registers (LFSRs)
% XA and XB with different tap configurations
% Code length: 10230 chips = 10.23 Mcps * 1 ms

% PRN-specific initial states for XB register (Table 3-III in IS-GPS-705)
% This is a simplified implementation - actual GPS L5 uses different method
XBInitial = [
    bin2dec('1010010111001'); % PRN 1
    bin2dec('1101111001100'); % PRN 2
    bin2dec('1111000011011'); % PRN 3
    bin2dec('1000110001111'); % PRN 4
    bin2dec('1011111111111'); % PRN 5
    bin2dec('1000100010011'); % PRN 6
    bin2dec('1111100010010'); % PRN 7
    bin2dec('1101010011110'); % PRN 8
    bin2dec('1101100001011'); % PRN 9
    bin2dec('1101100010101'); % PRN 10
    bin2dec('1001101001100'); % PRN 11
    bin2dec('1010000010110'); % PRN 12
    bin2dec('1000010000110'); % PRN 13
    bin2dec('1000110110111'); % PRN 14
    bin2dec('1001010000010'); % PRN 15
    bin2dec('1011100100011'); % PRN 16
    bin2dec('1110000010110'); % PRN 17
    bin2dec('1011010101000'); % PRN 18
    bin2dec('1000111110000'); % PRN 19
    bin2dec('1101001110101'); % PRN 20
    bin2dec('1110101100110'); % PRN 21
    bin2dec('1011100110001'); % PRN 22
    bin2dec('1000011011001'); % PRN 23
    bin2dec('1100011001100'); % PRN 24
    bin2dec('1010010001110'); % PRN 25
    bin2dec('1000111000100'); % PRN 26
    bin2dec('1000101100001'); % PRN 27
    bin2dec('1111010000111'); % PRN 28
    bin2dec('1111001000110'); % PRN 29
    bin2dec('1011001110001'); % PRN 30
    bin2dec('1001001101111'); % PRN 31
    bin2dec('1000111101110'); % PRN 32
];

% Initialize output
codeReplica = zeros(1, 10230);

% Initialize XA register (all ones for GPS L5)
XA = ones(1, 13);

% Initialize XB register with PRN-specific initial state
XBInit = XBInitial(PRN);
XB = zeros(1, 13);
for i = 1:13
    XB(i) = bitget(XBInit, 14-i);
end

% Generate 10230 chips
for chip = 1:10230
    % Output is XOR of the last bits of XA and XB
    codeReplica(chip) = xor(XA(13), XB(13));

    % XA feedback: taps at positions 9, 10, 12, 13
    % Polynomial: 1 + X^9 + X^10 + X^12 + X^13
    newXA = xor(xor(xor(XA(9), XA(10)), XA(12)), XA(13));

    % XB feedback: taps at positions 1, 3, 4, 6, 7, 8, 12, 13
    % Polynomial: 1 + X + X^3 + X^4 + X^6 + X^7 + X^8 + X^12 + X^13
    newXB = xor(xor(xor(xor(xor(xor(xor(XB(1), XB(3)), XB(4)), XB(6)), XB(7)), XB(8)), XB(12)), XB(13));

    % Shift registers
    XA = [newXA XA(1:12)];
    XB = [newXB XB(1:12)];
end

% Convert from 0/1 to -1/+1
zeroIndices = find(codeReplica(:)==0);
codeReplica(zeroIndices) = -1;
