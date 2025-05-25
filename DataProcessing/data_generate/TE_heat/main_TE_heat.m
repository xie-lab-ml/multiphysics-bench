% main
flag=0;
lenth=148;
pml=10;

tic;
number=1;

for NN=1:number
    while(flag==0)
        flag=1;

        path_Elliptic='ellipticcsv\';
        Elliptic=generate_Elliptic_TE_heat(path_Elliptic , NN);

        parm=[];
        parm.NN=NN;
        parm.Em=3e5;
        parm.f=4e9;
        parm.theta=pi/3;

        parm.h_heat=15;

        parm.Sigma_Si_coef=randi([1e11,3e11]);
        parm.Pho_Al=10*rand+10;

        parm.e_a=Elliptic(1);
        parm.e_b=Elliptic(2);
        parm.angle=Elliptic(3);

        disp(['calculating ', num2str(NN)]);
        try
            TE_heat(parm); % COMSOL
        catch
            flag=0;
            disp('Error');
        end

    end

    % export
    
    path_data=['data\' num2str(NN) '.csv'];
    path_Ez=['Ez\' num2str(NN) '.mat'];  % path_data=['data\comsol.csv'];
    path_T=['T\' num2str(NN) '.mat'];
    path_parm=['parm\' num2str(NN) '.mat'];
    
    %Data Save
    data = csvread(path_data, 9,2); 
    dat_data = data(:, :); 

    temp1=dat_data(:,1);
    export_T_pml=reshape(temp1,lenth,lenth);
    export_T=export_T_pml(pml+1:lenth-pml,pml+1:lenth-pml);
    temp2=dat_data(:,2);
    export_Ez_pml=reshape(temp2,lenth,lenth);
    export_Ez=export_Ez_pml(pml+1:lenth-pml,pml+1:lenth-pml);
    save(path_T,'export_T');
    save(path_Ez,'export_Ez');
    save(path_parm,'parm');


    flag=0;
end

disp([num2str(number), ' set took '  , num2str(toc), 's']);


figure (4)
imagesc(imag(export_Ez.'))
colormap('jet')
colorbar
axis xy

figure (3)
imagesc(export_T')
colormap('jet')
colorbar
axis xy

