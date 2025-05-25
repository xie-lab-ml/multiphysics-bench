%
% main_VA
flag=0;
lenth=128;

tic;
number=11000;
L=1.28;


for NN=1:number
    while(flag==0)
        flag=1;

        parm=[];
        parm.NN=NN;

        parm.rho_A = 500 + 400*rand ;  % [500,900]
        parm.rho_x0 = -0.01 + 0.02*rand;  % [-0.01,0.01]
        parm.rho_y0 = -0.01 + 0.02*rand;  % [-0.01,0.01]
        parm.rho_sigma = 5 + 15*rand;  % [5,20]

        disp(['calculating ', num2str(NN)]);
        try
            VA(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export
    path_data=['data\' num2str(NN) '.csv'];

    path_p_t=['p_t\' num2str(NN) '.mat'];
    path_Sxx=['Sxx\' num2str(NN) '.mat'];
    path_Sxy=['Sxy\' num2str(NN) '.mat'];
    path_Syy=['Syy\' num2str(NN) '.mat'];
    path_x_u=['x_u\' num2str(NN) '.mat'];
    path_x_v=['x_v\' num2str(NN) '.mat'];
    path_rho_water=['rho_water\' num2str(NN) '.mat'];

    path_parm=['parm\' num2str(NN) '.mat'];

    %Data Save
    data=csvread(path_data, 9,0);

    x_data = data(:, 1);
    y_data = data(:, 2);

    p_t = data(:, 3);
    Sxx = data(:, 4);
    Sxy = data(:, 5);
    Syy =  data(:, 6);
    x_u =  data(:, 7)*1e-3;
    x_v =  data(:, 8)*1e-3;
    rho_water =  data(:, 9);

    p_t(isnan(p_t))=0;
    Sxx(isnan(Sxx))=0;
    Sxy(isnan(Sxy))=0;
    Syy(isnan(Syy))=0;
    x_u(isnan(x_u))=0;
    x_v(isnan(x_v))=0;

    export_p_t=reshape(p_t,lenth,lenth).';
    export_Sxx=reshape(Sxx,lenth,lenth).';
    export_Sxy=reshape(Sxy,lenth,lenth).';
    export_Syy=reshape(Syy,lenth,lenth).';
    export_x_u=reshape(x_u,lenth,lenth).';
    export_x_v=reshape(x_v,lenth,lenth).';
    export_rho_water=reshape(rho_water,lenth,lenth).';

    save(path_p_t,'export_p_t');
    save(path_Sxx,'export_Sxx');
    save(path_Sxy,'export_Sxy');
    save(path_Syy,'export_Syy');
    save(path_x_u,'export_x_u');
    save(path_x_v,'export_x_v');
    save(path_rho_water,'export_rho_water');

    save(path_parm,'parm');


    flag=0;


end

disp([num2str(number), ' set took '  , num2str(toc), 's']);
