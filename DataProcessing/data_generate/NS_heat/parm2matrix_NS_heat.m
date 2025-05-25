
tic;
number=3000;

% inpupt Q_heat

Q_heat_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Q_heat\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Q_heat_all(i, :, :) = mat_obj.export_Q_heat;
end

range_allQ_heat(2)=max(Q_heat_all(:));
range_allQ_heat(1)=min(Q_heat_all(:));


savepath=['Q_heat\' , 'range_allQ_heat.mat'];
save(savepath,'range_allQ_heat','-mat')


%%
% outpupt max min u_u, u_v

u_u_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['u_u\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    u_u_all(i, :, :) = mat_obj.export_u_u;  
end

range_allu_u(2)=max(u_u_all(:));
range_allu_u(1)=min(u_u_all(:));
savepath=['u_u\', 'range_allu_u.mat'];
save(savepath,'range_allu_u','-mat')


u_v_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['u_v\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    u_v_all(i, :, :) = mat_obj.export_u_v;  
end

range_allu_v(2)=max(u_v_all(:));
range_allu_v(1)=min(u_v_all(:));
savepath=['u_v\', 'range_allu_v.mat'];
save(savepath,'range_allu_v','-mat')


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



disp(['Took ' , num2str(toc), ' s ', 'to normalize ' num2str(number), ' sets of data']);

