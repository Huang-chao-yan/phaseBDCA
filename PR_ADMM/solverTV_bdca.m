function [uPR,inerER,er]=solverTV_bdca(Y,x, A, At,Params,AtA,u0)

%% set up paras
[n1,n2,L]=size(Y);
lambda=Params.lambda;
gamma=Params.gamma;
c=Params.const;
maxDCA=Params.maxDCA;
nIter=Params.nIter;
tol=Params.tol;
tau=Params.tau;
beta0=10;
xi=0.5;
%% build A matrix
DtD = @(x) lap(x);
B = @(x) b_mat(x,n1,n2,c,gamma,AtA,DtD); 

%% initialization
% u0=HIO_image(Nx,Ny,L,Y,A,At,AtA,HIO_iter);
% u0=rand(n1,n2);
u=u0;
[px,py] = grad(u);
qx = zeros(size(px));
qy = zeros(size(py));

%% interation
for oit=1:maxDCA
    v=real(2*At(Y.*sign(A(u)))+c*u);
    F=@(u,uTV)lambda*sum(sum(uTV))+sum((abs(A(u))-Y).^2,[1,2,3]);
    for i=1:nIter
        %%%%%%%%%%%solve the subproblem u by chaoyan
        uold=u;
        b=v+div(qx,qy)+gamma*div(px,py);
        %u=conjgrad(u,b,500,1e-6,B);
        [u,flag] = cgs(B,b(:),1e-6,5);
        u = reshape(u,[n1,n2]);
        %%%%%%%%%%solve the subproblem p
        [dux,duy] = grad(u);
        ptx = (gamma*dux-qx)/gamma;
        pty = (gamma*duy-qy)/gamma;
        p_norm=sqrt(ptx.^2+pty.^2);
        px = sign(ptx).*max(0,p_norm-lambda/gamma);
        py = sign(pty).*max(0,p_norm-lambda/gamma);
        %%%%%%%%%%%%%%%%%%%%% update the dual variable q
        qx = qx-tau*gamma*(dux-px);
        qy = qy-tau*gamma*(duy-py);
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

function bx = b_mat(x,m,n,c,gamma,AtA,DtD)
xx = reshape(x,[m,n]);
bxx = 2*real(AtA(xx)) +c*xx+gamma*DtD(xx);
bx = bxx(:);
end