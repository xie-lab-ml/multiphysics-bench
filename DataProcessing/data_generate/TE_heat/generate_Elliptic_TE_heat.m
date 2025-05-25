function [Elliptic] = generate_Elliptic_TE_heat(path_Elliptic , NN)

% generate_Circle

parm_e_a=20*rand+10; % [20,30]
parm_e_b=10*rand+10; % [10,20]
parm_angle=360*rand; % [0,360]

Elliptic=[parm_e_a,parm_e_b,parm_angle];

writematrix(Elliptic, [path_Elliptic ,num2str(NN), '.csv']);

end
