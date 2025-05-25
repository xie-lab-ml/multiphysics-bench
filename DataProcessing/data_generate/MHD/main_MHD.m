%
% main_MHD

flag=0;
lenth=128;

tic;
number=1500;

% 
for NN=1:number
    while(flag==0)
        flag=1;

        parm=[];
        parm.NN=NN;

        parm.Br_A = rand ;  % [0,1]
        parm.Br_x0 = -0.01 + 0.02 * rand;  % [-0.01,0.01]
        parm.Br_y0 = -0.01 + 0.02 * rand;  % [-0.01,0.01]
        parm.Br_sigma = 0.2 + 0.8*rand;  % [0.2,1]
        
        disp(['calculating ', num2str(NN)]);
        try
            MHD(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export

    path_data=['data\' num2str(NN) '.csv'];

    path_Jx=['Jx\' num2str(NN) '.mat'];
    path_Jy=['Jy\' num2str(NN) '.mat'];
    path_Jz=['Jz\' num2str(NN) '.mat'];
    path_u_u=['u_u\' num2str(NN) '.mat'];
    path_u_v=['u_v\' num2str(NN) '.mat'];
    path_Br=['Br\' num2str(NN) '.mat'];
    path_parm=['parm\' num2str(NN) '.mat'];

    %Data Save
    %
    lines = readlines(path_data);

    x_start = find(contains(lines, '% Grid')) + 1;
    y_start = find(contains(lines, '% Grid')) + 2;
    Jx_start = find(contains(lines, '% mef.Jx')) + 1;
    Jy_start = find(contains(lines, '% mef.Jy')) + 1;
    Jz_start = find(contains(lines, '% mef.Jz')) + 1;
    u_start  = find(contains(lines, '% u (m/s)')) + 1;
    v_start  = find(contains(lines, '% v (m/s)')) + 1;
    Br_start  = find(contains(lines, '% A*exp(-((x-x0)^2 + (y-y0)^2)[1/cm^2]/(2*sigma^2))')) + 1;

    data = struct('x',[],'y',[],'Jx', [], 'Jy', [], 'Jz', [], 'u', [], 'v', [], 'Br', []);

    num_rows = 128;
    data.x = str2num(strjoin(lines(x_start : x_start + 1 - 1), ';'));
    data.y = str2num(strjoin(lines(y_start : y_start + 1 - 1), ';'));
    data.Jx = str2num(strjoin(lines(Jx_start : Jx_start + num_rows - 1), ';'));
    data.Jy = str2num(strjoin(lines(Jy_start : Jy_start + num_rows - 1), ';'));
    data.Jz = str2num(strjoin(lines(Jz_start : Jz_start + num_rows - 1), ';'));
    data.u  = str2num(strjoin(lines(u_start  : u_start  + num_rows - 1), ';'));
    data.v  = str2num(strjoin(lines(v_start  : v_start  + num_rows - 1), ';'));
    data.Br  = str2num(strjoin(lines(Br_start  : Br_start  + num_rows - 1), ';'));

    export_Jx = fliplr(data.Jx);
    export_Jy = fliplr(data.Jy);
    export_Jz = fliplr(data.Jz);
    export_u = fliplr(data.u);
    export_v = fliplr(data.v);
    export_Br = fliplr(data.Br);

    save(path_Jx,'export_Jx');
    save(path_Jy,'export_Jy');
    save(path_Jz,'export_Jz');
    save(path_u_u,'export_u');
    save(path_u_v,'export_v')
    save(path_Br,'export_Br')
    save(path_parm,'parm');

    flag=0;


    % figure;
    % imagesc(x_data,y_data,export_u_u);
    % xlabel('x');
    % ylabel('y');
    % zlabel('z');
    % colorbar; % 添加颜色条
    % axis xy
    % % % caxis([190,200])
    % axis equal;
    % % % axis tight;
    % colormap(jet);


end

disp([num2str(number), ' set took '  , num2str(toc), 's']);


