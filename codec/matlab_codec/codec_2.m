function [Tn] = codec_2(Y,Z)
%Y is 1d variable. Computes Tn(Y,Z)
%   Detailed explanation goes here

[n,q] = size(Z);
W = Z;

[Mbig,dMbig] = knnsearch(W,W,'k',3);

M = Mbig(:,2);

[~,p] = sort(Y,'descend');
R = 1:n;
R(p) = R;
clear p

L = flip(R); %basically this can be accomplished by doing 12-15 with 'ascend' instead of 'descend'
RM = R(M);

minRM = n*min(R,RM);

Tn_num = sum(minRM-L.^2)/n^2; %this does estimateconditionQ and returns Qn
Tn_den = sum(L.*(n-L))/n^2;

Tn = Tn_num/Tn_den;

end

