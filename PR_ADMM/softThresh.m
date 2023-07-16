function p=softThresh(p0,beta)
[~,n]=size(p0);n=n/2;
Z1=p0(:,1:n);
Z2=p0(:,n+1:2*n);
    V = abs(Z1).^2 + abs(Z2).^2;
    V = sqrt(V);
    
    V(V==0) = 1;
    
    V = max(V - beta, 0)./V;
    
    
    Y1 = Z1.*V;
    Y2 = Z2.*V;
    
    %Y1=Y1.*(1-Flag)+max(0,-beta).*Flag*sqrt(.5);
    %Y2=Y2.*(1-Flag)+max(0,-beta).*Flag*sqrt(.5);
 p=[Y1 Y2];   
end