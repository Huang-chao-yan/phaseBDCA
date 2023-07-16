function x=conjgrad(x,b,maxIt,tol,Ax_func)
%tic 
% conjgrad.m
%
%   Conjugate gradient optimization
%
%     written by Sunghyun Cho (sodomau@postech.ac.kr)
%
    
    r = b - Ax_func(x);
    p = r;
    rsold = sum(sum(r.*r));

    for iter=1:maxIt
        Ap = Ax_func(p);
        alpha = rsold/sum(sum(p.*Ap));
        x=x+alpha*p;
%         if exist('visfunc', 'var')
%             visfunc(x, iter, func_param);
%         end
        r=r-alpha*Ap;
        rsnew=sum(sum(r.*r));
        if sqrt(rsnew)<tol
            break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
   % toc
end
