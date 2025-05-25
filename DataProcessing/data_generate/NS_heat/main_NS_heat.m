%
% main_NS_heat
flag=0;
lenth=128;

tic;
number=11000;
L=0.128;


for NN=1:1
    while(flag==0)
        flag=1;

        path_Circle='NS_heat\circlecsv\';
        Circle=generate_Circle(L, path_Circle , NN);

        parm=[];
        parm.NN=NN;
        parm.centerx=Circle(1);
        parm.centery=Circle(2);
        parm.radius=Circle(3);

        parm.Q_heat=randi([50, 100]);

        disp(['calculating ', num2str(NN)]);
        try
            NS_heat(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export
    path_data=['data\' num2str(NN) '.csv'];

    path_Q_heat=['Q_heat\' num2str(NN) '.mat'];
    path_u_u=['u_u\' num2str(NN) '.mat'];
    path_u_v=['u_v\' num2str(NN) '.mat'];
    path_T=['T\' num2str(NN) '.mat'];
    path_parm=['parm\' num2str(NN) '.mat'];

    %Data Save
    data=csvread(path_data, 9,0);

    x_data = data(:, 1);
    y_data = data(:, 2);

    Q_heat = data(:, 3);
    u_u = data(:, 4);
    u_v = data(:, 5);
    T = data(:, 6);

    u_u(isnan(u_u))=0;
    u_v(isnan(u_v))=0;

    export_Q_heat=reshape(Q_heat,lenth,lenth)';
    export_u_u=reshape(u_u,lenth,lenth)';
    export_u_v=reshape(u_v,lenth,lenth)';
    export_T=reshape(T,lenth,lenth)';


    save(path_Q_heat,'export_Q_heat');
    save(path_u_u,'export_u_u');
    save(path_u_v,'export_u_v');
    save(path_T,'export_T');
    save(path_parm,'parm');


    flag=0;


end

disp([num2str(number), ' set took '  , num2str(toc), 's']);



