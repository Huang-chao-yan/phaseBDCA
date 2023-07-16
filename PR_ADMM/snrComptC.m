function [snr,scale,I]=snrComptC(I,uu)
%% uu as reference
scale=exp(-1i*angle(trace(uu'*I)));
%scale=sum(I'*uu)/norm(I,'fro')^2;
%scale=exp(-1i*angle(trace(uu(20:end-20,20:end-20)'*I(20:end-20,20:end-20))));


I=scale* I;
snr=-20*log10(norm(I - (uu), 'fro')/norm(I, 'fro'));

end