%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NN prediction
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

rng default
load('NN_Perf.mat')

%% Input parameters
LP = [300 400];

X_P = [
    % thicknesses (nm)
    [25 30]; % LH
    LP; % LP
    [25 30]; % LE

    % mobility (m^2/V/s)
    [1e-6 1e-6]; % muHh
    [5.5e-4 5.5e-4]; % muPh
    [5.5e-4 5.5e-4]; % muPe
    [1e-4 1e-4]; % muEe

    % density of state (1/m^3)
    [1e25 1e25]; % NvH
    [1e25 1e25]; % NcH
    [1e25 1e25]; % NvE
    [1e25 1e25]; % NcE
    [1e25 1e25]; % NvP
    [1e25 1e25]; % NcP

    % energy levels (eV)
    [5.3 5.3]; % chiHh
    [2.1 2.1]; % chiHe
    [5.5 5.5]; % chiPh
    [4 4]; % chiPe
    [6 6]; % chiEh
    [4.2 4.2]; % chiEe
    [4.3 4.3]; % Wlm
    [4.9 4.9]; % Whm

    % relative permittivity
    [3 3]; % epsH
    [6 6]; % epsP
    [3 3]; % epsE

    % average generation rate
    [260 260]/1.602e-19./(LP*1e-9); % G [1/m^3/s], calculated from i_sc [A/m^2]

    % recombination coefficients
    [1e-40 1e-40]; % Aug
    [1e-17 1e-17]; % Brad
    [1e-5 1e-5]; % Taue
    [1e-5 1e-5]; % Tauh
    [1e-25 1e-25]; % vII
    [1e-25 1e-25]; % vIII
    ];

X_P([4:13,26:31],:) = log10(X_P([4:13,26:31],:));

input = mapminmax('apply',X_P,PS_input);

%% Predict
tic
output = net(input);
toc

Perf = mapminmax('reverse',output,PS_output);
Voc = Perf(1,:)
FF = Perf(2,:)
PCE = Perf(3,:)