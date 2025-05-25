function [Elliptic] = generate_Elliptic(L, path_Elliptic , NN)  

    % generate_Circle 

    radius_a = (L/10) + (L/8 - L/10) * rand;
    radius_b = (L/10) + (L/8 - L/10) * rand;

    centerx = (-L/8) + (2*L/8) * rand;
    centery = (-L/8) + (2*L/8) * rand;

    Elliptic=[centerx,centery,radius_a,radius_b];

    writematrix(Elliptic, [path_Elliptic ,num2str(NN), '.csv']);

end
