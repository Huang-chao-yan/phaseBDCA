% %  Chris Metzler.
% % Nov 17 2017
function [x_hat]=GS(y,A,Apinv,iters)
%x, y, and z can be matrices, where each column represents a different problem.
x_hat=Apinv*(y.*exp(2*pi*rand(size(y))));
for i=1:iters
    z=y.*exp(1i*angle(A*x_hat));
    x_hat=Apinv*z;
end
end