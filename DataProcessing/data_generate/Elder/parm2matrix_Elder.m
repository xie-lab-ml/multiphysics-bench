% 初始化
num_samples = 1000;
time_steps = 11; % 

% 
range_S_c_t = inf * ones(time_steps, 2); % [min, max] for each time step
range_u_u_t = inf * ones(time_steps, 2);
range_u_v_t = inf * ones(time_steps, 2);
range_c_flow_t = inf * ones(time_steps, 2);

range_S_c_t(:,2) = -inf; 
range_u_u_t(:,2) = -inf; 
range_u_v_t(:,2) = -inf; 
range_c_flow_t(:,2) = -inf; 

% 
for i = 1:num_samples
    % 
    for t = 0:time_steps-1
        % 
        path_Sc_t = fullfile('S_c', num2str(i), [num2str(t) '.mat']);
        tmp = load(path_Sc_t);
        Sc_t = getfield(tmp, char(fieldnames(tmp)));
        range_S_c_t(t+1, 1) = min(range_S_c_t(t+1, 1), min(Sc_t(:))); % 
        range_S_c_t(t+1, 2) = max(range_S_c_t(t+1, 2), max(Sc_t(:))); % 
    end

    % 
    var_names = {'u_u', 'u_v', 'c_flow'};
    for var_idx = 1:3
        folder = var_names{var_idx};
        % 
        for t = 0:time_steps-1
            path_var_t = fullfile(folder, num2str(i), [num2str(t) '.mat']);
            tmp = load(path_var_t);
            data = getfield(tmp, char(fieldnames(tmp)));
            eval(sprintf('range_%s_t(t+1, 1) = min(range_%s_t(t+1, 1), min(data(:)));', folder, folder)); % 
            eval(sprintf('range_%s_t(t+1, 2) = max(range_%s_t(t+1, 2), max(data(:)));', folder, folder)); % 
        end
    end
end


%
save(fullfile('S_c', 'range_S_c_t.mat'), 'range_S_c_t');
save(fullfile('u_u', 'range_u_u_t.mat'), 'range_u_u_t');
save(fullfile('u_v', 'range_u_v_t.mat'), 'range_u_v_t');
save(fullfile('c_flow', 'range_c_flow_t.mat'), 'range_c_flow_t');


%% 
% 
range_u_u_t_999 = zeros(time_steps, 2);
range_u_v_t_99 = zeros(time_steps, 2);
range_c_flow_t_99 = zeros(time_steps, 2);

%
for t = 0:time_steps-1
    % u_u
    u_u_t_data = [];
    for i = 1:num_samples
        path_u_u_t = fullfile('u_u', num2str(i), [num2str(t) '.mat']);
        tmp = load(path_u_u_t);
        u_u_t_data = [u_u_t_data; getfield(tmp, char(fieldnames(tmp)))];
    end
    range_u_u_t_999(t+1, 1) = prctile(u_u_t_data(:), 0.3);
    range_u_u_t_999(t+1, 2) = prctile(u_u_t_data(:), 99.997);

    % u_v
    u_v_t_data = [];
    for i = 1:num_samples
        path_u_v_t = fullfile('u_v', num2str(i), [num2str(t) '.mat']);
        tmp = load(path_u_v_t);
        u_v_t_data = [u_v_t_data; getfield(tmp, char(fieldnames(tmp)))];
    end
    range_u_v_t_99(t+1, 1) = prctile(u_v_t_data(:), 0.01);
    range_u_v_t_99(t+1, 2) = prctile(u_v_t_data(:), 99.7);
    
    % c_flow
    c_flow_t_data = [];
    for i = 1:num_samples
        path_c_flow_t = fullfile('c_flow', num2str(i), [num2str(t) '.mat']);
        tmp = load(path_c_flow_t);
        c_flow_t_data = [c_flow_t_data; getfield(tmp, char(fieldnames(tmp)))];
    end
    range_c_flow_t_99(t+1, 1) = prctile(c_flow_t_data(:), 0.3);
    range_c_flow_t_99(t+1, 2) = prctile(c_flow_t_data(:), 99.7);
end

% 
save(fullfile('u_u', 'range_u_u_t_999.mat'), 'range_u_u_t_999');
save(fullfile('u_v', 'range_u_v_t_99.mat'), 'range_u_v_t_99');
save(fullfile('c_flow', 'range_c_flow_t_99.mat'), 'range_c_flow_t_99');


%%
range_allS_c = [];
range_allS_c(1) = min(range_S_c_t(:));
range_allS_c(2) = max(range_S_c_t(:));

range_allu_u = [];
range_allu_u(1) = min(range_u_u_t(:));
range_allu_u(2) = max(range_u_u_t(:));

range_allu_v = [];
range_allu_v(1) = min(range_u_v_t(:));
range_allu_v(2) = max(range_u_v_t(:));

range_allc_flow = [];
range_allc_flow(1) = min(range_c_flow_t(:));
range_allc_flow(2) = max(range_c_flow_t(:));


save(fullfile('S_c', 'range_allS_c.mat'), 'range_allS_c');
save(fullfile('u_u', 'range_allu_u.mat'), 'range_allu_u');
save(fullfile('u_v', 'range_allu_v.mat'), 'range_allu_v');
save(fullfile('c_flow', 'range_allc_flow.mat'), 'range_allc_flow');