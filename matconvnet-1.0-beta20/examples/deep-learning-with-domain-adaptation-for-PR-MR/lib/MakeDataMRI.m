function imdb = MakeDataMRI(imdb_ft, ds, cpN)
N           = size(imdb_ft, 1);
VIEW        = size(imdb_ft, 2);
VIEW_dw     = VIEW/ds;
NFRAME      = size(imdb_ft, 3);

%%
[sino_ft, sino_ang]         = radial_downsampling(imdb_ft,1);
[sino_ft_dw, sino_ang_dw]	= radial_downsampling(sino_ft,ds);

%%
if nargin < 3
    cpN     = 2^(nextpow2(N) + 1);
end

dfN         = floor((cpN - N)/2);

exP         = size(radon(zeros(cpN), 1), 1);
dfP         = floor((exP - cpN)/2);

sino_ft     = padarray(sino_ft, [dfN, 0], 'pre');
sino_ft     = padarray(sino_ft, [dfN + mod(cpN, 2), 0], 'post');

sino_ft_dw  = padarray(sino_ft_dw, [dfN, 0], 'pre');
sino_ft_dw	= padarray(sino_ft_dw, [dfN + mod(cpN, 2), 0], 'post');

sino     	= abs(ifftshift(ifft(fftshift(sino_ft, 1), [], 1), 1))*exP^2;
sino_dw     = abs(ifftshift(ifft(fftshift(sino_ft_dw, 1), [], 1), 1))*exP^2;

sino        = padarray(sino, [dfP, 0], 'pre');
sino        = padarray(sino, [dfP + mod(exP, 2), 0], 'post');

sino_dw   	= padarray(sino_dw, [dfP, 0], 'pre');
sino_dw   	= padarray(sino_dw, [dfP + mod(exP, 2), 0], 'post');

orig        = 0;
orig_dw     = 0;

labels      = [];
data        = [];


for iframe = 1:NFRAME
    
    theta               = sino_ang(:, iframe);
    theta_dw            = sino_ang_dw(:, iframe);
    
    labels(:,:,iframe)	= iradon(sino(:,:, iframe), theta, cpN, 'cosine');
    data(:,:,iframe)    = iradon(sino_dw(:,:, iframe), theta_dw, cpN, 'cosine');
    
end

labels(labels < 0)  = 0;
orig(orig < 0)      = 0;

imdb.images.labels  = single(labels);
imdb.images.data	= single(data);
imdb.images.orig    = single(orig);
imdb.images.orig_dw = single(orig_dw);
imdb.images.sino    = sino;
imdb.images.sino_dw = sino_dw;
imdb.images.theta  	= sino_ang;
imdb.images.theta_dw= sino_ang_dw;

imdb.meta.frame    	= NFRAME;
imdb.meta.imageSize = cpN;
imdb.meta.view      = VIEW;
imdb.meta.view_dw   = VIEW_dw;

return ;

