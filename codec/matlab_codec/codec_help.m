clear all,close all,clc
d_y = 1;%assume 1d for now
q = 1;

%% Vignette 1
n = 10000;
p = 3;
X = rand(n,p);
Y = mod(sum(X,2),1);
codec_2(Y,X(:,1))
codec_2(Y,X(:,2))
codec_2(Y,X(:,3))

codec_2(Y,X(:,1:2))

codec_2(Y,X)

codec_3(X(:,1:2),Y,X(:,3))

codec_3(X(:,1),Y,X(:,3))

%% Vignette 2
n = 1000;
p = 2;
X = randn(n,p); %data matrix with each row to be (xi,yi,zi)
Y = sum(X.^2,2);

Z = atan(X(:,1)./X(:,2));
codec_2(Y,Z) % reproduces bottom example in https://cran.r-project.org/web/packages/FOCI/vignettes/codec.html
codec_3(X(:,1),Y,Z) % reproduces bottom example in https://cran.r-project.org/web/packages/FOCI/vignettes/codec.html
