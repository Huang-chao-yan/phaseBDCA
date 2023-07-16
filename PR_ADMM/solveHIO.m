function [u_HIO,error,snr]=solveHIO(A,At,AtA,PS,PM,I,YGauss,iter_max,tol,u_ini,delta)
z_ini=A(u_ini);
% u=u_ini;
z=z_ini;
for kk=1:iter_max
    z=(1+delta)*PS(PM(z))+z-PS(z)-delta*PM(z);  
    u= real(At(z)./AtA); 
    I_tmp=exp(-1i*angle(trace(u'*I)))*I;
    error(kk)=norm(u-I_tmp,2);
    if error(kk)<tol
        break
    end
end
u_HIO=u;
snr=snrCompt(u_HIO,I);
end