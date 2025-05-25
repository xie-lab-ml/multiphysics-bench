% 
tic;
number=10000;

% inpupt kappa

kappa_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['kappa\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    kappa_all(i, :, :) = mat_obj.export_kappa;
end

range_allkappa(2)=max(kappa_all(:));
range_allkappa(1)=min(kappa_all(:));


savepath=['kappa\' , 'range_allkappa.mat'];
save(savepath,'range_allkappa','-mat')


%%
% outpupt max min ec_V, u_flow, v_flow

ec_V_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['ec_V\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    ec_V_all(i, :, :) = mat_obj.export_ec_V;  
end

range_allec_V(2)=max(ec_V_all(:));
range_allec_V(1)=min(ec_V_all(:));
savepath=['ec_V\', 'range_allec_V.mat'];
save(savepath,'range_allec_V','-mat')

u_flow_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['u_flow\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    u_flow_all(i, :, :) = mat_obj.export_u_flow;  
end

range_allu_flow(2)=max(u_flow_all(:));
range_allu_flow(1)=min(u_flow_all(:));
savepath=['u_flow\', 'range_allu_flow.mat'];
save(savepath,'range_allu_flow','-mat')


v_flow_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['v_flow\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    v_flow_all(i, :, :) = mat_obj.export_v_flow;  
end

range_allv_flow(2)=max(v_flow_all(:));
range_allv_flow(1)=min(v_flow_all(:));
savepath=['v_flow\', 'range_allv_flow.mat'];
save(savepath,'range_allv_flow','-mat')


disp(['Took ' , num2str(toc), ' s ', 'to normalize ' num2str(number), ' sets of data']);