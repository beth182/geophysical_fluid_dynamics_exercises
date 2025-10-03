% ==============================================================================
% GFD_ex1_problem_4
%
% Lena Pfister
% ==============================================================================

clear all;

%Specify path for plots
figpath = './GFD_ex1_problem_4/'; 

% initial conditions
X0 = 0.;
Y0 = 1.;
Z0 = 0.;

% some parameters
p = 10;
beta = 8/3;
r = 0.5;
dt = 0.01;

% time vector
t = [0 : dt : 100];
nt = length(t);

% ==============================================================================
% Runge-Kutta integration
% ==============================================================================
    function [X,Y,Z] = rk4_lorenz(X0,Y0,Z0,p,beta,r,dt,nt);
    X(1) = X0;
    Y(1) = Y0;
    Z(1) = Z0;
    for it = 2 : nt;
      % TASK: add the equations for dYdt, dZdt, Y1, and Z1
      dXdt = p*(Y(it-1)-X(it-1));
    %  dYdt = 
    %  dZdt = 
      X1 = X(it-1) + dXdt*dt/2;
    %  Y1 = 
    %  Z1 = 

      % TASK: add the equations for dYdt, dZdt, Y2, and Z2
      dXdt = p*(Y1-X1);
    %  dYdt = 
    %  dZdt = 
      X2 = X(it-1) + dXdt*dt/2;
    %  Y2 = 
    %  Z2 = 

      % TASK: calculate steps 3 and 4

      % TASK: add the equations for X(it), Y(it), and Z(it)
    %  X(it) = 
    %  Y(it) = 
    %  Z(it) = 
    end
    end



% ====== time integration ====================================================
[X,Y,Z] = rk4_lorenz(X0,Y0,Z0,p,beta,r,dt,nt);
  

% ====== Plotting results ====================================================
hfig = figure('Position',[100 100 700 300], 'PaperPositionMode','auto');
hax = axes('Position',[0.10 0.20 0.85 0.70]); hold on; box on;
set(hax, 'TickDir','out', 'FontSize',12, 'XLim',[0 max(t)*0.75]);
title(['r = ',num2str(r)]);
xlabel('Time');
hp(1) = plot(t, X, 'k', 'LineWidth',1.2);
hp(2) = plot(t, Y, 'g--', 'LineWidth',1.2);
hp(3) = plot(t, Z, 'r', 'LineWidth',1.2);
legend(hp, {'X','Y','Z'}, 'FontSize',12, 'Location','NorthEast', 'Orientation','horizontal');
legend('boxoff');
print(hfig, '-depsc', [figpath,'r',num2str(r),'.eps']);

if abs(r-28.0)<1e-5
  % butterfly
  hfig = figure('Position',[100 100 700 700], 'PaperPositionMode','auto');
  hax = axes('Position',[0.10 0.10 0.85 0.85]); hold on; box on;
  set(hax, 'TickDir','out', 'FontSize',12);
  plot3(X, Y, Z, 'k');
  view(30,30);
  xlabel('x'); ylabel('y'); zlabel('z');
  print(hfig, '-depsc', [figpath,'butterfly.eps']);
end


