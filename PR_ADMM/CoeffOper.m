function gamma=CoeffOper(op,alpha,beta)
% This subroutine implement alpha op beta = gamma, where op= + - = * s n
Level=length(alpha);
[nD,nD1]=size(alpha{1});
for ki=1:Level
    for ji=1:nD
        for jj=1:nD
            if op=='='
                gamma{ki}{ji,jj}=alpha{ki}{ji,jj};
            elseif op=='-'
                gamma{ki}{ji,jj}=alpha{ki}{ji,jj}-beta{ki}{ji,jj};
            elseif op=='+'
                gamma{ki}{ji,jj}=alpha{ki}{ji,jj}+beta{ki}{ji,jj};
            elseif op=='*'
                gamma{ki}{ji,jj}=alpha{ki}{ji,jj}*beta;
            elseif op=='s'
                gamma{ki}{ji,jj}=wthresh(alpha{ki}{ji,jj},'s',beta{ki}{ji,jj});
            end
        end
    end
end