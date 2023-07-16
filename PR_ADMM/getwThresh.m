function muLevel=getwThresh(mu,wLevel,Level,D);
nfilter=1;
nD=length(D);
if wLevel<=0
    for ki=1:Level
        for ji=1:nD-1
            for jj=1:nD-1
                muLevel{ki}{ji,jj}=mu*nfilter*norm(D{ji})*norm(D{jj});
            end
        end
        nfilter=nfilter*norm(D{1});
    end
else
    for ki=1:Level
        for ji=1:nD-1
            for jj=1:nD-1
                muLevel{ki}{ji,jj}=mu*nfilter;
            end
        end
        nfilter=nfilter*wLevel;
    end
end