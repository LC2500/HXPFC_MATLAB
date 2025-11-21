% 2D Navier Stokes Staggered Grid Lid Driven Flow
clear
clc
close all

%% ----------- Phase Field Crystal Parameters ----------- %%
% For triagular phase, psi_0 = 0.06 sig = 0.12
psi_0 = 0.06; % initial density field 
s = 0.12; % Temperature like parameter
a = 0.5; 
b = 0.33;
M = 1; % Diffusion Coefficient or Mobility
eps = 0;
v = 0;
sc = 1; % scaling effect on mobility
M = M/sc^2;
p1 = 1; p2 = 1;
m = 1; kBT = 1; 

%% Grid-Discretization 
My = 2^10; Mx = My;
My_seg = (My-1); Mx_seg = (Mx-1);
Ly = 128*(2/sqrt(3)); Lx = (Mx_seg/My_seg)*Ly; % Could be needed for PFC 
dx = 1/Mx_seg; dy = 1/My_seg;
dx2 = dx^2; dy2 = dy^2;

x = linspace(0,1,Mx); y = linspace(0,1,My);
[X,Y] = meshgrid(x,y); % X - columns Y - Rows 
Nx = Mx/2; Ny = My/2;
 
r_arr = sqrt(X.^2 + Y.^2);

kx = (2*pi/Lx)*[0:Nx 1-Nx:-1]'; ky = (2*pi/Ly)*[0:Ny 1-Ny:-1]';
k2 = kx.^2 + (ky.^2)'; % Laplacian Calculation
[kX,kY] = meshgrid(kx,ky);
k_arr = sqrt(kX.^2 + (kY.^2));

ff_factor = (Lx*Ly)/(Mx_seg*My_seg); % prefactor for rectangular rule
V = (Lx*Ly);
ff_factor = ff_factor/V; % Volume normalized pre-factor

%% Mirrored Convolution Parameters
kx_m = (pi/Lx)*[0:Mx 1-Mx:-1]'; % Changed, removed division by Lx
ky_m = (pi/Ly)*[0:My 1-My:-1]'; % Changed, removed division by Ly
[kX_m,kY_m] = meshgrid(kx_m,ky_m);
k_arr_m = sqrt(kX_m.^2 + (kY_m.^2));

%% Time Variables 
t = 0.0065; dt = 0.000001;
N_t_steps = t/dt;
t_smple = 100;

%% Correlation Function 
% Greenwood et al. PRL 105, 045702 (2010)
% Huan et al. Computational Materials Science 122 (2016) 167–176
% Correlation Function Parameters 
k_10 = 2*pi; k_11 = 2*pi*sqrt(2);
rho_10 = 1; rho_11 = 1/sqrt(2);
B_10 = 4; B_11 = 4;
al_10 = 1; al_11 = 1;
a_tri = 2/sqrt(3); 
a_sq = 1;

c2_k_f = @(k) exp(-s^2*k_10^2/(2*rho_10*B_10)).*exp(-(k-k_10).^2/(2*al_10.^2)) + exp(-s^2*k_11^2/(2*rho_11*B_11)).*exp(-(k-k_11).^2/(2*al_11.^2));

figure(1)
c2_k = c2_k_f(k_arr);
contourf(kX,kY,c2_k)
colorbar

c2_km = c2_k_f(k_arr_m);

%% PFC Initialization
in_fil = exp(-((X-0.5073).^2+(Y-0.5038).^2)/0.00015); % Initial Condition Middle of Domain 

psi = readmatrix("tri_s_0.12_psi_0_0.06_Lx=147.txt")';
psi = in_fil.*psi + psi_0;

figure(2)
imagesc(x,y,psi)
colorbar
% colormap("bone")
xlabel('x')
ylabel('y')

psi_n = psi;
psi_np1 = zeros(My,Mx);
d_mu_dx = zeros(Mx,My);
d_mu_dy = zeros(Mx,My);

convl = non_PBC_convl_FDM_2(psi_n',c2_km);
mu_n =  psi_n - a*psi_n.^2 + b*psi_n.^3 - p2*convl';

%% ----------- Navier-Stokes Parameters ----------- %%

%% Velocity Parameters
v_top = 1; v_ns = 0; % No-slip = 0

%% Error Tolerance for Pressure Poisson
tol = 0.0000001; % 0.0001; 
w = 1; % Parameter for Pressure Poisson

%% Initialization on Staggered Grid, including ghost cells
% u_n = zeros(Ny+1,Nx);
% Uncomment if only top-plate driven flow
% u_n = v_top*((Y-dy/2));
% u_n = [u_n; v_top*((1+dy/2))*ones(1,Mx)];

% Uncomment if top and bottom driven flow
u_n = v_top*(2*(Y-dy/2)-1);
u_n = [u_n; v_top*(2*(1+dy/2)-1)*ones(1,Mx)];

v_n = zeros(My,Mx+1);
u_np12 = zeros(My+1,Mx);
v_np12 = zeros(My,Mx+1);

% X Boundary Conditions 
% No-Flow Normal Boundary Conditions on RHS and LHS 
% u_n(:,1) = 0; u_n(:,end) = 0;
% Lid Driven Cavity
u_n(end,:) = 2*v_top - u_n(end-1,:);
% No-slip Boundary Conditions on bottom 
% u_n(1,:) = 2*v_ns - u_n(2,:); % Uncomment if top-plate driven
u_n(1,:) = -2*v_top - u_n(2,:); % Uncomment if top-bottom plate driven

% Y boundary Conditions
% % No-slip Boundary Conditions on sides
% v_n(:,1) = 2*v_ns - v_n(:,2);
% v_n(:,end) = 2*v_ns - v_n(:,end-1);
% PBC Boundary Conditions on sides
v_n(:,Mx+1) = v_n(:,2); v_n(:,1) = v_n(:,Mx);
% No-Flow Normal Boundary Conditions on RHS and LHS
v_n(1,:) = 0; v_n(end,:) = 0;

P_n = zeros(My+1,Mx+1);

u_avg_n = (u_n(2:end,:) + u_n(1:end-1,:))/2;
v_avg_n = (v_n(:,2:end) + v_n(:,1:end-1))/2;

%% ----------- Dimensionless Parameter Estimation ----------- %% 
Re = 100; ooRe = 1/Re;
Eu = 1; ooEu = 1/Eu;
PI_2 = 20; % Peclet Like Number
PI_3 = 1; % Euler-Like Navier Stokes Number

%% PFC and N-S Grid Parameters %%
lam = (dt/dx^2)/PI_2;
gam = dt/dx;
lam_NS = ooRe*dt/dx2;

%% Write Initial Conditions %% 
write_vtk(0,Lx,Lx,Ly,Mx,My,psi_n)

%% ----------- Main Time Loop ----------- %% 
for i = 1:N_t_steps
    conv = (1-u_avg_n*gam-v_avg_n*gam); 
    vx_1 = u_avg_n*gam; vy_1 = v_avg_n*gam;

    psi_np1(2:My-1,2:Mx-1) = (conv(2:My-1,2:Mx-1).*psi_n(2:My-1,2:Mx-1) + ...
        vx_1(2:My-1,2:Mx-1).*psi_n(2:My-1,1:Mx-2) + ...
        vy_1(2:My-1,2:Mx-1).*psi_n(1:My-2,2:Mx-1) + ...
        lam*(mu_n(2:My-1,3:Mx)+ mu_n(3:My,2:Mx-1) + ...
            mu_n(2:My-1,1:Mx-2) + mu_n(1:My-2,2:Mx-1) - ...
                4*(mu_n(2:My-1,2:Mx-1))));
    
    % PBC in X 
    psi_np1(2:My-1,1) = (conv(2:My-1,1).*psi_n(2:My-1,1) + ...
        vx_1(2:My-1,1).*psi_n(2:My-1,Mx-1) + ...
        vy_1(2:My-1,1).*psi_n(1:My-2,1) + ...
        lam*(mu_n(2:My-1,2)+mu_n(3:My,1) + ...
            mu_n(2:My-1,Mx-1) + mu_n(1:My-2,1) - ...
                4*(mu_n(2:My-1,1))));
    psi_np1(2:My-1,Mx) = psi_np1(2:My-1,1);

    % NBC in Y 
    psi_np1(1,2:Mx-1) = (conv(1,2:Mx-1).*psi_n(1,2:Mx-1) + ...
        vx_1(1,2:Mx-1).*psi_n(1,1:Mx-2) + ...
        vy_1(1,2:Mx-1).*psi_n(2,2:Mx-1) + ...
        lam*(mu_n(1,3:Mx) + 2*mu_n(2,2:Mx-1) + ...
            mu_n(1,1:Mx-2) - ...
                4*(mu_n(1,2:Mx-1))));

    psi_np1(My,2:Mx-1) = (conv(My,2:Mx-1).*psi_n(My,2:Mx-1) + ...
        vx_1(My,2:Mx-1).*psi_n(My,1:Mx-2) + ...
        vy_1(My,2:Mx-1).*psi_n(My-1,2:Mx-1) + ...
        lam*(mu_n(My,3:Mx) + 2*mu_n(My-1,2:Mx-1) + ...
            mu_n(My,1:Mx-2) - ...
                4*(mu_n(My,2:Mx-1))));

    % Corner Nodes (1,1) (1,Mx) (My,Mx) (My,Mx)
    psi_np1(1,1) = (conv(1,1).*psi_n(1,1) + ...
        vx_1(1,1).*psi_n(1,Mx-1) + ...
        vy_1(1,1).*psi_n(2,1) + ...
        lam*(mu_n(1,2) + 2*mu_n(2,1) + ...
            mu_n(1,Mx-1) - ...
                4*(mu_n(1,1))));

    psi_np1(1,Mx) = (conv(1,Mx).*psi_n(1,Mx) + ...
        vx_1(1,Mx).*psi_n(1,Mx-1) + ...
        vy_1(1,Mx).*psi_n(2,Mx) + ...
        lam*(mu_n(1,2) + 2*mu_n(2,Mx) + ...
            mu_n(1,Mx-1) - ...
                4*(mu_n(1,Mx))));

    psi_np1(My,1) = (conv(My,1).*psi_n(My,1) + ...
        vx_1(My,1).*psi_n(My,Mx-1) + ...
        vy_1(My,1).*psi_n(My-1,Mx) + ...
        lam*(mu_n(My,2) + 2*mu_n(My-1,1) + ...
            mu_n(1,Mx-1) - ...
                4*(mu_n(My,1))));

    psi_np1(My,Mx) = (conv(My,Mx).*psi_n(My,Mx) + ...
        vx_1(My,Mx).*psi_n(My,Mx-1) + ...
        vy_1(My,Mx).*psi_n(My-1,Mx) + ...
        lam*(mu_n(My,2) + 2*mu_n(My-1,Mx) + ...
            mu_n(1,Mx-1) - ...
                4*(mu_n(My,Mx))));
    
    psi_n = psi_np1;
    convl = non_PBC_convl_FDM_2(psi_n',c2_km);
    psi_2 = psi_n.^2;
    psi_3 = psi_2.*psi_n;
   
    mu_n =  psi_n - a*psi_2 + b*psi_3 - p2*convl';
    
    % Multiply momentum coupling by "kBT/m"
    % Parameters for Momentum Coupling xPFC to Navier-Stokes 
    d_mu_dx(:,2:Mx-1) = (mu_n(:,3:Mx)-mu_n(:,1:Mx-2))/(2*dx);
    d_mu_dx(:,Mx) = (mu_n(:,2)-mu_n(:,Mx-1))/(2*dx);
    d_mu_dx(:,1) = d_mu_dx(:,Mx);
    dmudx_avg_y = (d_mu_dx(2:My,:) + d_mu_dx(1:My-1,:))/2; 

    d_mu_dy(2:My-1,:) = (mu_n(3:My,:) - mu_n(1:My-2,:))/(2*dy);
    d_mu_dy(My,:) = (mu_n(My,:) - mu_n(My-1,:))/(dy);
    d_mu_dy(1,:) = (mu_n(2,:) - mu_n(1,:))/(dy);
    dmudy_avg_x = (d_mu_dy(:,2:Mx) + d_mu_dx(:,1:Mx-1))/2;

    psi_avg_y = (psi_n(2:My,:) + psi_n(1:My-1,:))/2; % Changed psi->psi_n 12/03/24
    psi_avg_x = (psi_n(:,2:Mx) + psi_n(:,1:Mx-1))/2; % Changed psi->psi_n 12/03/24
    
    % Step 1. Intermediate Velocity Update
    
    % Momentum Diffusivity u
    dif_un = ooRe*( (u_n(2:My,3:Mx) - 2*u_n(2:My,2:Mx-1) + u_n(2:My,1:Mx-2))/dx2 ...
    + (u_n(3:My+1,2:Mx-1) - 2*u_n(2:My,2:Mx-1) + u_n(1:My-1,2:Mx-1))/dy2 );
    
    % Convection u
    un_av2ip1 = (0.5*(u_n(:,2:Mx-1)+u_n(:,3:Mx))).^2; % VERIFIED
    un_av2i1 = (0.5*(u_n(:,1:Mx-2)+u_n(:,2:Mx-1))).^2; % VERIFIED 

    % Averaging over the extra dimension
    uv_kp1 = (( u_n(2:My,:) + u_n(3:My+1,:))/2).*((v_n(2:My,1:Mx) + v_n(2:My,2:Mx+1))/2); % VERIFIED
    uv_km1 = (( u_n(2:My,:) + u_n(1:My-1,:))/2).*((v_n(1:My-1,1:Mx) + v_n(1:My-1,2:Mx+1))/2); % VERIFIED
    
    convx_un = (un_av2ip1(2:My,:) - un_av2i1(2:My,:))/dx; % Only want interior Nodes VERIFEID
    convy_un = (uv_kp1(:,2:Mx-1)-uv_km1(:,2:Mx-1))/dy; % Only want interior Nodes VERIFIED
    
    % Non-Divergence Free X-Velocity Update 
    u_np12(2:My,2:Mx-1) = u_n(2:My,2:Mx-1) + dt*(dif_un - convx_un - convy_un + PI_3*psi_avg_y(:,2:Mx-1).*dmudx_avg_y(:,2:Mx-1) );
    
    % Periodic Boundary Conditions at x = Lx and x = 0
    % Convection u
    convx_PBC_1 = ((0.5*(u_n(2:My,Mx)+u_n(2:My,2))).^2 - (0.5*(u_n(2:My,Mx-1)+u_n(2:My,Mx))).^2)/dx; % VERIFIED 
    convy_PBC_2 = (uv_kp1(:,Mx)-uv_km1(:,Mx))/dy; % VERIFIED
    dif_PBC = ooRe*( (u_n(2:My,2) - 2*u_n(2:My,Mx) + u_n(2:My,Mx-1))/dx2 ...
    + (u_n(3:My+1,Mx) - 2*u_n(2:My,Mx) + u_n(1:My-1,Mx))/dy2 );
    u_np12(2:My,Mx) = u_n(2:My,Mx) + dt*(dif_PBC- convx_PBC_1 - convy_PBC_2 + PI_3*psi_avg_y(:,Mx).*dmudx_avg_y(:,Mx));
    u_np12(2:My,1) = u_np12(2:My,Mx);

    % Lid Driven 
    u_np12(end,:) = 2*v_top - u_np12(end-1,:);
    % No-slip Boundary Conditions on bottom 
    % u_np12(1,:) = 2*v_ns - u_n(2,:); % Uncomment if top-plate driven
    u_np12(1,:) = -2*v_top - u_np12(2,:); % Uncomment if top-bottom plate driven

     % Momentum Diffusion v
    dif_vn = ooRe*( (v_n(2:My-1,3:Mx+1) - 2*v_n(2:My-1,2:Mx) + v_n(2:My-1,1:Mx-1))/dx2 ...
    + (v_n(3:My,2:Mx) - 2*v_n(2:My-1,2:Mx) + v_n(1:My-2,2:Mx))/dy2 );

    % Convection v
    vn_av2kp1 = (0.5*(v_n(2:My-1,:)+v_n(3:My,:))).^2; % VERIFIED
    vn_av2k1 = (0.5*(v_n(1:My-2,:)+v_n(2:My-1,:))).^2; % VERIFIED

    % Averaging over the extra dimension 
    uv_jp1 = (1/4)*(u_n(1:My,2:Mx)+u_n(2:My+1,2:Mx)).*(v_n(:,3:Mx+1)+v_n(:,2:Mx)); % VERIFIED
    uv_jm1 = (1/4)*(u_n(1:My,1:Mx-1)+u_n(2:My+1,1:Mx-1)).*(v_n(:,1:Mx-1)+v_n(:,2:Mx)); % VERIFIED
    
    convx_vn = (vn_av2kp1(:,2:Mx) - vn_av2k1(:,2:Mx))/dx; % Only accessing interior nodes VERIFIED
    convy_vn = (uv_jp1(2:My-1,:)-uv_jm1(2:My-1,:))/dy; % Only accessing interior nodes VERIFIED

    % Non-Divergence Free Y-Velocity Update 
    v_np12(2:My-1,2:Mx) = v_n(2:My-1,2:Mx) + dt*(dif_vn - convx_vn - convy_vn + PI_3*psi_avg_x(2:My-1,:).*dmudy_avg_x(2:My-1,:));
    % Intermediate Y boundary Conditions
    % PBC Boundary Conditions on sides
    v_np12(:,Mx+1) = v_np12(:,2); v_np12(:,1) = v_np12(:,Mx);
    % No-Flow Normal Boundary Conditions on top and bottom
    v_np12(1,:) = 0; v_np12(end,:) = 0;

    % Step 2. Pressure Poisson Problem
    div = (u_np12(2:end-1,2:end)-u_np12(2:end-1,1:end-1))/dx + ...
        (v_np12(2:end,2:end-1) - v_np12(1:end-1,2:end-1))/dy; % cell 
    % centered derivatives
    PP_rhs = ooEu*div/dt;

    er = 1;
    tic
    while er > tol
        P_np1 = zeros(size(P_n));
%         P_np1(2:Ny,2:Nx) = (1/4)*(P_np1(2:Ny,1:Nx-1) + P_n(2:Ny,3:Nx+1) ...
%         + P_np1(1:Ny-1,2:Nx) + P_n(3:Ny+1,2:Nx) - dx2*PP_rhs);
        P_np1(2:My,2) = (1/4)*(P_np1(2:My,1) + P_n(2:My,3) ...
         + P_np1(1:My-1,2) + P_n(3:My+1,2) - dx2*PP_rhs(:,1));
        ct = 2;
        for ii = 3:Mx
            P_np1(2:My,ii) = P_n(2:My,ii) + w*((1/4)*(P_np1(2:My,ii-1) + P_n(2:My,ii+1) ...
         + P_np1(1:My-1,ii) + P_n(3:My+1,ii) - dx2*PP_rhs(:,ct)) - P_n(2:My,ii));
            ct = ct+1;
        end
        
        % Neumann Boundary Conditions in Y 
        P_np1(1,:) = P_np1(2,:);
        P_np1(end,:) = P_np1(end-1,:);
        
        % Periodic Boundary Conditions at x = Lx and x = 0
        P_np1(:,Mx+1) = P_np1(:,2);
        P_np1(:,1) = P_np1(:,Mx);

%         P_np1(:,end) = P_np1(:,end-1);
%         P_np1(:,1) = P_np1(:,2);
      
        er = sqrt(sum((P_n(:) - P_np1(:)).^2));
        P_n = P_np1;
    end
    toc

%     tic
%         P_np1 = solve_PP(L_pp,L,U,PP_rhs,dx2,Mx,My);
%     toc

    P_np1 = P_n;
    dP_dx = (P_np1(:,2:end) - P_np1(:,1:end-1))/dx;
    dP_dy = (P_np1(2:end,:) - P_np1(1:end-1,:))/dy;

    u_np1 = u_np12 - (dt*Eu)*dP_dx;
    v_np1 = v_np12 - (dt*Eu)*dP_dy;
    u_n = u_np1; v_n = v_np1;
    

    % PBCS...

    % X Boundary Conditions 
    % Periodic Boundary Conditions at x = Lx and x = 0
    % Lid Driven Cavity
    u_n(end,:) = 2*v_top - u_n(end-1,:);
    % No-slip Boundary Conditions on bottom 
    % u_n(1,:) = 2*v_ns - u_n(2,:); % Uncomment if top-plate driven
    u_n(1,:) = -2*v_top - u_n(2,:); % Uncomment if top-bottom plate driven

    % Y boundary Conditions
    % Periodic Boundary Conditions at x = Lx and x = 0
    % PBC Boundary Conditions on sides
    v_n(:,Mx+1) = v_n(:,2); v_n(:,1) = v_n(:,Mx);
    % No-Flow Normal Boundary Conditions on RHS and LHS
    v_n(1,:) = 0; v_n(end,:) = 0;


    % Returning to collocated grid to watch study
    u_avg_n = (u_n(2:end,:) + u_n(1:end-1,:))/2;
    v_avg_n = (v_n(:,2:end) + v_n(:,1:end-1))/2;
    
    tmp = mod(i,t_smple);
    if tmp == 0 
        write_vtk(i,Lx,Lx,Ly,Mx,My,psi_n)
        % write_VF(i-1,0,Lx,Ly,Mx,My,u_avg_n,v_avg_n)
    end

end

%% Analytical Solution for Error Estimate
% u_ana = v_top*(Y/Ly);
u_ana = v_top*(2*Y-1);
er_u = abs(u_avg_n-u_ana);
write_VF(i,0,Lx,Ly,Mx,My,u_avg_n,v_avg_n)

