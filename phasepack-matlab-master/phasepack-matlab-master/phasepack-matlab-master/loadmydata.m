function [A,b]=loadmydata(imageSize)
load('b')
load('omega')
load('init1')
init=init1;
II=ones(imageSize);
[m,n]=size(II);
s=.5;t=.5;
x=exp(1i*2*pi*s*[0:n-1]/n);
y=exp(1i*2*pi*t*[0:m-1]/m);
[x,y]=meshgrid(x,y);
D=x.*y;

FT = p2DFT(ones(size(II)), size(II), 1, 2);     % from sparseMRI_v0.2
A0 = abs(FT * II);
A1 = abs(FT * (II+   D.*II));
A2 = abs(FT * (II-1i*D.*II));
A=[A0;A1;A2];
bb0=ifftshift(b0.*Omega);
bb1=ifftshift(b1.*Omega);
bb2=ifftshift(b2.*Omega);
z0(Omega)=bb0(Omega).*exp(1i*2*pi*init(:,1));
z1(Omega)=bb1(Omega).*exp(1i*2*pi*init(:,2));
z2(Omega)=bb2(Omega).*exp(1i*2*pi*init(:,3));

b=[z0;z1;z2];
