function u_ER=ER_image(r,c,A,At,AtA,PM,PS,iter_max)
% % Y is our obervation.
% beta = 0.9;
% %% Creating initial complex-valued field distribution at the detector plane
% % phase = zeros(N,N,L);
% phase = (2*rand(r,c,L) - 1)*pi;
% field_detector_0 = Y.*exp(1i*phase);
% 
% object_0 = At(field_detector_0)./AtA;
% gk = real(object_0);
% %% Iterative loop
% for ii = 1:Iteration
%     
% field_detector = A(gk);
% % Replacing updated amplitude for measured amplitude
% field_detector_updated = Y.*exp(1i*angle(field_detector));  
% 
% % Getting updated object distribution
% gk_prime = real(At(field_detector_updated)./AtA);
% %error(ii) = norm(gk_prime - x, 'fro');
% 
% % Object constraint
% index_p=gk_prime>0;
% index_np=gk_prime<=0;
% gk(index_p)=gk_prime(index_p);
% gk(index_np)=gk(index_np)-beta*gk_prime(index_np);
% 
% end
u_ini=rand(r,c)+1i*rand(r,c);
z_ini=A(u_ini);
% u=u_ini;
z=z_ini;
for kk=1:iter_max
    z=PS(PM(z));  

end
%     u= real(At(z)./AtA); 
    u= real(At(z)./AtA);
    u_ER=u;
end

