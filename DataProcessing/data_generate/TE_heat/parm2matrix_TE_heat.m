tic;
number = 3000;

% input materials
for i = 1:3000
    datamater = zeros(128);
    path_elliptic = ['ellipticcsv\' num2str(i) '.csv'];
    elliptic = importdata(path_elliptic); 

    path_parm = ['parm\' num2str(i) '.mat'];
    parm = importdata(path_parm);

    e_a = elliptic(1); 
    e_b = elliptic(2); 
    angle = elliptic(3); 
    angle = deg2rad(angle);
    center_x = 0; 
    center_y = 0;

    % a1
    for j = 1:128
        for k = 1:128
            x0 = -63.5 + j - 1;
            y0 = -63.5 + k - 1;

            x_rot = (x0 - center_x) * cos(angle) + (y0 - center_y) * sin(angle);
            y_rot = -(x0 - center_x) * sin(angle) + (y0 - center_y) * cos(angle);

            if ((x_rot / e_a)^2 + (y_rot / e_b)^2 <= 1)
                datamater(j, k) = parm.Sigma_Si_coef;
            else
                datamater(j, k) = parm.Pho_Al;
            end
        end
    end

    mater = datamater;
    savepath = ['mater\' num2str(i) '.mat'];
    save(savepath, 'mater', '-mat');
end


%%
% outpupt max min EZ T
Ez_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Ez\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Ez_all(i, :, :) = mat_obj.export_Ez;
end

max_abs_Ez=max(abs(Ez_all(:)));
savepath=['Ez\' , 'max_abs_Ez.mat'];
save(savepath,'max_abs_Ez','-mat')


T_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['T\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    T_all(i, :, :) = mat_obj.export_T;  
end

range_allT(2)=max(T_all(:));
range_allT(1)=min(T_all(:));
savepath=['T\', 'range_allT.mat'];
save(savepath,'range_allT','-mat')
