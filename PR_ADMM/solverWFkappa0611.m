function [uPR,inerER,er]=solverWFkappa0611(Y,x, A, At,Params,AtA,u0,lambda)

%% set up paras
[n1,n2,~]=size(Y);
%lambda=4550;%for sigma=30;
%remember to change eta, it is the same value with lambda
%1150;%for sigma=10
%1e3;
gamma=Params.gamma;
c=Params.const;
maxDCA=Params.maxDCA;
nIter=Params.nIter;
tol=Params.tol;
tau=1;%Params.tau;
beta0=10;%10
xi=0.5;%0.5
eta=1400;
%4150;%1150;
%% build A matrix
DtD=ones(size(u0));
%DtD = @(x) lap(x);
B = @(x) b_mat(x,n1,n2,c,gamma,AtA,eta,DtD); 

%% initialization
% u0=HIO_image(Nx,Ny,L,Y,A,At,AtA,HIO_iter);
% u0=rand(n1,n2);
u=u0;
% [px,py] = grad(u);
% qx = zeros(size(px));
% qy = zeros(size(py));
%W
%frame parameters
 frame=1;
 Level=2;
 wLevel=1/2;
% initialization
[Dw,R]=GenerateFrameletFilter(frame);
nD=length(Dw);
p=FraDecMultiLevel(u,Dw,Level);
q=p;

t=u;
d=zeros(size(t));

%% interation
for oit=1:maxDCA
    v=real(2*At(Y.*sign(A(u)))+c*u);
    F=@(u,uTV)lambda*sum(sum(uTV))+sum((abs(A(u))-Y).^2,[1,2,3]);
    for i=1:nIter
        %%%%%%%%%%%solve the subproblem u
        uold=u;
        b=v+FraRecMultiLevel(q,R,Level)+1.1*gamma*FraRecMultiLevel(p,R,Level)+eta*t-d;
        %u=conjgrad(u,b,500,1e-6,B);
        [u,flag] = cgs(B,b(:),1e-6,5);
        u = reshape(u,[n1,n2]);
        %%%%%%%%%%solve the subproblem p
        du=FraDecMultiLevel(u,Dw,Level);
       for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
%                      pt{ki}{ji,jj} = du{ki}{ji,jj}-q{ki}{ji,jj}/gamma;
                     p{ki}{ji,jj}=softThresh(du{ki}{ji,jj}-q{ki}{ji,jj}/gamma,lambda/gamma);
                end
            end
       end    
       
       %%%%% t
       t=min(1,max(0,u+d/eta));
       
        %%%%%%%%%%%%%%%%%%%%% update the q
        for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                    q{ki}{ji,jj}= q{ki}{ji,jj}-tau*gamma*(du{ki}{ji,jj}-p{ki}{ji,jj});
                end
            end
       end    
        %%%%%%%%%%%%%%%  update the d
        d=d+tau*eta*(u-t);
        
      
        %%%%%
        if (norm(u-uold, 'fro')/norm(uold,'fro')<tol)
            break;
        end
        inerER(i)=norm(u-uold, 'fro')/norm(uold,'fro');
    end
%% linear search 
 y=u;
 dk=y-uold;
if (norm(dk,'fro')/norm(uold,'fro')<tol)
    break;
elseif (beta0>0) 
        beta=beta0;       
        [yx_beta,yy_beta]=grad(y+beta.*dk);
        y_betaTV=sqrt(abs(yx_beta).^2+abs(yy_beta).^2);
        [yx,yy]=grad(y);
        y_TV=sqrt(abs(yx).^2+abs(yy).^2);
      if (F(y+beta.*dk,y_betaTV)-F(y,y_TV)+tau*beta*(norm(dk,'fro')^2)>0)
          beta=xi*beta;  
      else
          beta=beta0;
      end
      u_new=y+beta.*dk;
 if (norm(u_new-uold, 'fro')/norm(uold,'fro')<tol)  
            break;
  end
end
%%
er(oit)=norm(x - exp(-1i*angle(trace(x'*u))) * u, 'fro');
end
uPR=real(exp(-1i*angle(trace(x'*u))) * u);
end

function [dux,duy] = grad(u)
dux = [diff(u,1,2), u(:,1)-u(:,end)];
duy = [diff(u,1,1); u(1,:)-u(end,:)];
end

function dtxy = div(x,y)
dtxy = [x(:,end)-x(:,1),-diff(x,1,2)];
dtxy = dtxy + [y(end,:)-y(1,:);-diff(y,1,1)];
end

function dtdu = lap(u)
[dux,duy] = grad(u);
dtdu = div(dux,duy);
end

function bx = b_mat(x,m,n,c,gamma,AtA,eta,DtD)
xx = reshape(x,[m,n]);
bxx = 2*real(AtA(xx)) +(c+gamma-eta)*xx;
bx = bxx(:);
end