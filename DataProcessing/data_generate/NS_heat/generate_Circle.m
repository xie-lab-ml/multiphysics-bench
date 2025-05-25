function [Circle] = generate_Circle(L, path_Circle , NN)  

    % generate_Circle - 生成一个随机圆

    % 随机生成圆的半径（范围在 L/10 到 L/8 之间）

    radius = (L/10) + (L/8 - L/10) * rand;

    % 随机生成圆心的 x 和 y 坐标（范围在 -L/8 到 L/8 之间）
    centerx = (-L/8) + (2*L/8) * rand;
    centery = (-L/8) + (2*L/8) * rand;

    Circle=[centerx,centery,radius];

    writematrix(Circle, [path_Circle ,num2str(NN), '.csv']);

end
