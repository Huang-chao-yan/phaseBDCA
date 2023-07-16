function snr=snrCompt(I,uu)
I=I(:);
uu=uu(:);
%snr=-20*log10(norm(I - real(uu), 'fro')/norm(real(uu), 'fro'));
%snr=-20*log10(norm(I - (uu), 'fro')/norm(uu, 'fro'));
snr=-20*log10(norm(I - (uu), 'fro')/norm(I, 'fro'));

end