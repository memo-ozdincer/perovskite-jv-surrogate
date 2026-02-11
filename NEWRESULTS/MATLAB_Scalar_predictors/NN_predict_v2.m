%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% NN prediction (for .txt input)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

rng default
load('NN_Perf.mat')
load('LHS_parameters_m.txt')

%% Processing input
e_c = 1.60217657e-19; 
X_P = LHS_parameters_m;
isc = e_c*X_P(:,2)*1e-9.*X_P(:,25);
X_P(:,[4:13,26:31]) = log10(X_P(:,[4:13,26:31]));
input = mapminmax('apply',X_P.',PS_input);

%% Predict
tic
output = net(input);
toc

Perf = (mapminmax('reverse',output,PS_output)).';
Voc = Perf(:,1);
FF = Perf(:,2);
PCE = Perf(:,3);

%% Save data
writematrix(isc,'NN_predicted_isc.txt');
writematrix(Voc,'NN_predicted_Voc.txt');
writematrix(FF,'NN_predicted_FF.txt');
writematrix(PCE,'NN_predicted_PCE.txt');
T = table(isc,Voc,FF,PCE);
writetable(T,'NN_predicted_Perf.txt');