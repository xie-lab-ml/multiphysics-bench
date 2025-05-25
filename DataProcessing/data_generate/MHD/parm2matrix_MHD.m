% 
tic;
number=10000;

% inpupt Br

Br_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Br\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Br_all(i, :, :) = mat_obj.export_Br;
end

range_allBr(2)=max(Br_all(:));
range_allBr(1)=min(Br_all(:));


savepath=['Br\' , 'range_allBr.mat'];
save(savepath,'range_allBr','-mat')


%%
% outpupt max min Jx, Jy, Jz, u_u, u_v

Jx_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Jx\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Jx_all(i, :, :) = mat_obj.export_Jx;  
end

range_allJx(2)=max(Jx_all(:));
range_allJx(1)=min(Jx_all(:));
savepath=['Jx\', 'range_allJx.mat'];
save(savepath,'range_allJx','-mat')


Jy_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Jy\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Jy_all(i, :, :) = mat_obj.export_Jy;  
end

range_allJy(2)=max(Jy_all(:));
range_allJy(1)=min(Jy_all(:));
savepath=['Jy\', 'range_allJy.mat'];
save(savepath,'range_allJy','-mat')


Jz_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['Jz\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    Jz_all(i, :, :) = mat_obj.export_Jz;  
end

range_allJz(2)=max(Jz_all(:));
range_allJz(1)=min(Jz_all(:));
savepath=['Jz\', 'range_allJz.mat'];
save(savepath,'range_allJz','-mat')


u_u_all = zeros(number, 128, 128);

file_paths = cell(1, number);

for i = 1:number
    file_paths{i} = ['u_u\' num2str(i) '.mat'];
    mat_obj = matfile(file_paths{i});
    u_u_all(i, :, :) = mat_obj.export_u;  
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
    u_v_all(i, :, :) = mat_obj.export_v;  
end

range_allu_v(2)=max(u_v_all(:));
range_allu_v(1)=min(u_v_all(:));
savepath=['u_v\', 'range_allu_v.mat'];
save(savepath,'range_allu_v','-mat')


disp(['Took ' , num2str(toc), ' s ', 'to normalize ' num2str(number), ' sets of data']);