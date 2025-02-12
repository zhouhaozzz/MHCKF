% read data
elec_heater = cell(1, 11);
for i = 1:10
    f1 = sprintf('data/%d.txt', i);
    data = load(f1);
    elec_heater{i} = data;
end

f1 = 'data/heat source.txt';
data = load(f1);
elec_heater{11} = data;

% Convert cell array to 3D array and scale
start_time = tic;
elec_heater = cell2mat(reshape(elec_heater, 1, 1, [])) * 1e9;

datax = elec_heater(:, 1, 11);
init_face = elec_heater(:, 2, 11);
heat_face = permute(elec_heater(:, 2, 1:10), [1 3 2]);

K = 0 - init_face;

% 使用线性最小二乘法求解各个加热片的功率
lb = zeros(size(heat_face, 2), 1); % lower bounds
ub = Inf(size(heat_face, 2), 1); % upper bounds


result = lsqlin(heat_face, K, [], [], [], [], lb, ub);
elapsed_time = toc(start_time);

% 输出各个加热片的功率
P = result;
clc
fprintf('Elapsed time: %f seconds\n', elapsed_time);
%disp('Optimal Heating Powers:');
%disp(P);

% 验证结果
T_computed = heat_face * P;
%%
figure;
plot(datax, init_face + T_computed, 'DisplayName', 'Optimized surface');
hold on;
plot(datax, init_face, 'DisplayName', 'Initial surface');
legend;
xlabel('x [mm]');
ylabel('Deformation [nm]');
hold off;

figure;
plot(datax, init_face + T_computed, 'DisplayName', 'Optimized surface');
legend;
xlabel('x [mm]');
ylabel('Deformation [nm]');

figure;
plot(P * 0.001, 'DisplayName', 'Optimized surface');
legend;
xlabel('electric heater');
ylabel('Heat Flux [W/mm^2]');
%%
lb = -Inf(size(heat_face, 2), 1); % lower bounds
ub = Inf(size(heat_face, 2), 1); % upper bounds
result = lsqlin(heat_face, K, [], [], [], [], lb, ub);
% 输出各个加热片的功率
P = result;
disp('Optimal Heating Powers:');
disp(P);

