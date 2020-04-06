%	By Philip Torr 2002
%	copyright Microsoft Corp.

function [c,ceq] = torr_nonlcon_f2x2(f, nx1,ny1,nx2,ny2, m3)
%c = ...     % Compute nonlinear inequalities at f.
%ceq = ...   % Compute nonlinear equalities at f.

%g(1) = norm(f) -1.0;
%g(2) = f(1) * (f(5) * f(9) - f(6) * f(8)) - f(2) * (f(4) * f(9) - f(6) * f(7)) + f(3) * (f(4) * f(8) - f(5) * f(7));
c = [];
%what norm should we use!
ceq(1) = sqrt(f(1)^2 + f(2)^2 + f(4)^2 + f(5)^2)- 1;
%ceq(1)= norm(f) -1.0;
ceq(2) = f(1) * (f(5) * f(9) - f(6) * f(8)) - f(2) * (f(4) * f(9) - f(6) * f(7)) + f(3) * (f(4) * f(8) - f(5) * f(7));
