% 
tic;
number=10000;

% inpupt rho_water

rho_water_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['rho_water\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    rho_water_all(i, :, :) = mat_obj.export_rho_water;
end

range_allrho_water(2)=max(rho_water_all(:));
range_allrho_water(1)=min(rho_water_all(:));


savepath=['rho_water\' , 'range_allrho_water.mat'];
save(savepath,'range_allrho_water','-mat')


%%
%
% main_VA
tic;
lenth=128;

number=10000;

omega = pi*1e5;
rho_AL=2730;

for NN=10001:11000

    % export
    path_data=['data\' num2str(NN) '.csv'];
    path_x_u=['x_u\' num2str(NN) '.mat'];
    path_x_v=['x_v\' num2str(NN) '.mat'];

    %Data Save
    data=csvread(path_data, 9,0);
    x_u =  data(:, 7)*1e-3 * omega^2 * rho_AL;
    x_v =  data(:, 8)*1e-3 * omega^2 * rho_AL;

    x_u(isnan(x_u))=0;
    x_v(isnan(x_v))=0;

    export_x_u=reshape(x_u,lenth,lenth).';
    export_x_v=reshape(x_v,lenth,lenth).';

    save(path_x_u,'export_x_u');
    save(path_x_v,'export_x_v');

end

disp([num2str(number), ' set took '  , num2str(toc), 's']);


imagesc(imag(export_x_v))
colorbar



%%
tic;

% vars = {'p_t', 'Sxx', 'Sxy', 'Syy','x_u', 'x_v'};
vars = {'x_u', 'x_v'};

for k = 1:length(vars)
    var_name = vars{k};
    real_all = zeros(number, 128, 128);
    imag_all = zeros(number, 128, 128);
    
    for i = 1:number
        file_path = [var_name, '\', num2str(i), '.mat'];
        export_data = matfile(file_path).(['export_', var_name]);
        real_all(i, :, :) = real(export_data);
        imag_all(i, :, :) = imag(export_data);
    end
    
    eval(['range_allreal_', var_name, ' = [min(real_all(:)), max(real_all(:))];']);
    eval(['range_allimag_', var_name, ' = [min(imag_all(:)), max(imag_all(:))];']);
    
    save([var_name, '\range_allreal_', var_name, '.mat'], ['range_allreal_', var_name]);
    save([var_name, '\range_allimag_', var_name, '.mat'], ['range_allimag_', var_name]);
end

disp(['Took ' , num2str(toc), ' s ', 'to normalize ' num2str(number), ' sets of data']);


%
% imagesc(real(export_Syy));
% colorbar




