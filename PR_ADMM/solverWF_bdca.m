function [uPR,inerER,er]=solverWF_bdca(Y,x, A, At,Params,AtA,u0)

%% set up paras
[n1,n2,L]=size(Y);
lambda=1e3;%Params.lambda;%1e3;
gamma=Params.gamma;
c=Params.const;
maxDCA=Params.maxDCA;
nIter=Params.nIter;
tol=Params.tol;
tau=Params.tau;
beta0=10;
xi=0.5;



%% build A matrix
%DtD = @(x) lap(x);
DtD=ones(size(x));
B = @(x) b_mat(x,n1,n2,c,gamma,AtA,DtD); 

%% initialization
% u0=HIO_image(Nx,Ny,L,Y,A,At,AtA,HIO_iter);
% u0=rand(n1,n2);
u=u0;
%% W
%frame parameters
 frame=1;
 Level=2;
 wLevel=1/2;
% initialization
[Dw,R]=GenerateFrameletFilter(frame);
nD=length(Dw);
p=FraDecMultiLevel(u,Dw,Level);
q=p;

% [px,py] = grad(u);
% qx = zeros(size(px));
% qy = zeros(size(py));

%% interation
for oit=1:maxDCA
    v=real(2*At(Y.*sign(A(u)))+c*u);
    F=@(u,uTV)lambda*sum(sum(uTV))+sum((abs(A(u))-Y).^2,[1,2,3]);
    for i=1:nIter
        %%%%%%%%%%%solve the subproblem u
        uold=u;
%         b=v+div(qx,qy)+gamma*div(px,py);
        b=v+FraRecMultiLevel(q,R,Level)+gamma*FraRecMultiLevel(p,R,Level);
        [u,flag] = cgs(B,b(:),1e-6,5);
        u = reshape(u,[n1,n2]);
        %%%%%%%%%%solve the subproblem p
%         [dux,duy] = grad(u);
        du=FraDecMultiLevel(u,Dw,Level);
%         ptx = (gamma*dux-qx)/gamma;
%         pty = (gamma*duy-qy)/gamma;

%         pt=(gamma*du-q)/gamma;
       for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                    pt{ki}{ji,jj}=(gamma*du{ki}{ji,jj}-q{ki}{ji,jj})/gamma;
                    p{ki}{ji,jj}=softThresh(pt{ki}{ji,jj},lambda/gamma);
                end
            end
       end  
%         p_norm=sqrt(ptx.^2+pty.^2);
%         px = sign(ptx).*max(0,p_norm-lambda/gamma);
%         py = sign(pty).*max(0,p_norm-lambda/gamma);
        %%%%%%%%%%%%%%%%%%%%% update the dual variable q
        
       for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                    q{ki}{ji,jj}=q{ki}{ji,jj}-tau*gamma*(du{ki}{ji,jj}-p{ki}{ji,jj});
              
                end
            end
       end 
%         qx = qx-tau*gamma*(dux-px);
%         qy = qy-tau*gamma*(duy-py);
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
bxx = 2*real(AtA(xx)) +c*xx+gamma*xx;%DtD(xx);
bx = bxx(:);
end