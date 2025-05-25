%
% main_Elder
flag=0;
lenth=128;

tic;
number=1010;
L=1.28;


for NN=1011:1100
    while(flag==0)
        flag=1;

        parm=[];
        parm.NN=NN;

        parm.Sc_A = 1e-3 + 7e-3*rand ;  % [1e-3,8e-3]
        parm.Sc_x0 = -70 + 140*rand;  % [-70,70]
        parm.Sc_y0 = -30 + 60*rand;  % [-30,30]
        parm.Sc_sigma = 10 + 60*rand;  % [10,50]
        

        disp(['calculating ', num2str(NN)]);
        try
            Elder(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export
    % path_data=['data\' num2str(NN) '.csv'];

    path_data=['data\data.csv'];

    path_parm=['parm\' num2str(NN) '.mat'];
    save(path_parm,'parm');

    data=csvread(path_data, 9,0);
    x_data = data(:, 1);
    y_data = data(:, 2);


    for T = 0:10
        path_u_u_T = fullfile('u_u', num2str(NN), [num2str(T), '.mat']);
        path_u_v_T = fullfile('u_v', num2str(NN), [num2str(T), '.mat']);
        path_c_flow_T = fullfile('c_flow', num2str(NN), [num2str(T), '.mat']);
        path_S_c_T = fullfile('S_c', num2str(NN), [num2str(T), '.mat']);

        %
        u_u = data(:, 3 + T * 4);
        u_v = data(:, 4 + T * 4);
        c_flow = data(:, 5 + T * 4);
        S_c = data(:, 6 + T * 4);

        %
        export_u_u = reshape(u_u, lenth, lenth)';
        export_u_v = reshape(u_v, lenth, lenth)';
        export_c_flow = reshape(c_flow, lenth, lenth)';
        export_S_c = reshape(S_c, lenth, lenth)';

        %
        folder_u_u = fileparts(path_u_u_T);
        folder_u_v = fileparts(path_u_v_T);
        folder_c_flow = fileparts(path_c_flow_T);
        folder_S_c = fileparts(path_S_c_T);

        if ~exist(folder_u_u, 'dir')
            mkdir(folder_u_u);
        end
        if ~exist(folder_u_v, 'dir')
            mkdir(folder_u_v);
        end
        if ~exist(folder_c_flow, 'dir')
            mkdir(folder_c_flow);
        end
        if ~exist(folder_S_c, 'dir')
            mkdir(folder_S_c);
        end

        % 
        save(path_u_u_T, 'export_u_u');
        save(path_u_v_T, 'export_u_v');
        save(path_c_flow_T, 'export_c_flow');
        save(path_S_c_T, 'export_S_c');
    end

    flag=0;

end

disp([num2str(number), ' set took '  , num2str(toc), 's']);
