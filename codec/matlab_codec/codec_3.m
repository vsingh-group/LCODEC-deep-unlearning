function [Tn] = codec_3(X,Y,Z)
%Y is 1d variable computes Tn(Y,Z|X)
%   Note that Y is second argument here vs codec_2 since there we assume X
%   is not available...
[n,px] = size(X);


W = [X,Z];

[Nbig,dNbig] = knnsearch(X,X,'k',3);
[Mbig,dMbig] = knnsearch(W,W,'k',3);

N = Nbig(:,2);
M = Mbig(:,2);

[~,p] = sort(Y,'descend');
R = 1:n;
R(p) = R;
clear p

RM = R(M);
RN = R(N);

minRM = min(R,RM);
minRN = min(R,RN);

Tn_num = sum(minRM-minRN)/n^2; %this does estimateconditionQ and returns Qn
Tn_den = sum(R-minRN)/n^2;

Tn = Tn_num/Tn_den;

end

