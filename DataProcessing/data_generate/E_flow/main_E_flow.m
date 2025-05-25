%
% main_EOF
flag=0;
lenth=128;

tic;
number=10;
L=1.28;


for NN=1:number
    while(flag==0)
        flag=1;

        path_Elliptic='ellipticcsv\';
        Elliptic=generate_Elliptic(L, path_Elliptic , NN);

        parm=[];
        parm.NN=NN;
        parm.centerx=Elliptic(1);
        parm.centery=Elliptic(2);
        parm.radius_a=Elliptic(3);
        parm.radius_b=Elliptic(4);

        parm.kappa0 = 2e-3 + 2e-2*rand ;  % [2e-3,2.2e-2]
        parm.kappa_x0 = 0.5 * rand;  % [0,0.5]
        parm.kappa_y0 = 0.5 * rand;  % [0,0.5]
        parm.kappa_sigma = 0.2 + 0.8*rand;  % [0.2,1]

        parm.kappa_x0=parm.kappa_x0*1e-3;   % mm
        parm.kappa_y0=parm.kappa_y0*1e-3; 
        parm.kappa_sigma=parm.kappa_sigma*1e-3;

        disp(['calculating ', num2str(NN)]);
        try
            E_flow(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export
    path_data=['data\' num2str(NN) '.csv'];

    path_ec_V=['ec_V\' num2str(NN) '.mat'];
    path_u_flow=['u_flow\' num2str(NN) '.mat'];
    path_v_flow=['v_flow\' num2str(NN) '.mat'];
    path_kappa=['kappa\' num2str(NN) '.mat'];
    path_parm=['parm\' num2str(NN) '.mat'];

    %Data Save
    % path_data=['E:\DATA\EOF\Untitled.csv'];
    data=csvread(path_data, 9,0);

    x_data = data(:, 1);
    y_data = data(:, 2);

    ec_V = data(:, 3);
    u_flow = data(:, 4);
    v_flow = data(:, 5);
    kappa =  data(:, 6);

    ec_V(isnan(ec_V))=0;
    u_flow(isnan(u_flow))=0;
    v_flow(isnan(v_flow))=0;
    kappa(isnan(kappa))=0;

    export_ec_V=reshape(ec_V,lenth,lenth)';
    export_u_flow=reshape(u_flow,lenth,lenth)';
    export_v_flow=reshape(v_flow,lenth,lenth)';
    export_kappa=reshape(kappa,lenth,lenth)';

    save(path_ec_V,'export_ec_V');
    save(path_u_flow,'export_u_flow');
    save(path_v_flow,'export_v_flow');
    save(path_kappa,'export_kappa');
    save(path_parm,'parm');


    flag=0;


end

disp([num2str(number), ' set took '  , num2str(toc), 's']);
