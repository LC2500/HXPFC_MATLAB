function convl = non_PBC_convl_FDM_2(np,p_m)
% Convolves a non-periodic function with a periodic function in two
% dimensions 
% The non-periodic direction is assumed to be in the vertical directionL
r = height(np); % Number of rows

%% Debugging grid parameters for plotting

% p_m = c2f(k_arr);
% p_m = c2f;
%% Mirroring and Convolution
np_m = [np np(:,end:-1:1)];
np_m = [np_m; np_m];
% p_m = [p;p];
% p_m = [p_m p_m];

% contourf(X,Y,np_m)
% 
% contourf(X,Y,np_m);
% contourf(kX_m,kY_m,p_m);

convl_m = real(ifft2( fft2(np_m) .* p_m ) );

% contourf(kX_m,kY_m,convl_m)
convl = convl_m(1:r,1:r);

end