%%
clear ;
close all;

restoredefaultpath();

addpath('lib');

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

%% CNN PARAMETER

networkType     = 'simplenn';

method          = 'residual';

solver_handle	= [];

imageRange      = [0, 0.1];
imageSize       = [512, 512, 1];

inputSize       = [256, 256, 1];

wgt             = 5e3;

%
numEpochs       = 1e3;

batchSize       = 2;
subbatchSize    = 2;
numSubBatches   = ceil(batchSize/subbatchSize);
batchSample     = 1;

lrnrate         = logspace(-3, -5, 1e3);
wgtdecay        = 1e-4;

meanNorm        = false;
varNorm         = false;

gpus            = 1;
train           = struct('gpus', gpus);


dataDir         = './data/';
netDir          = './network/';

db              = 1;


%% PR PARAMETER
pr.niter          	= 1e2;
pr.th             	= 0;
pr.wname           	= 'db3';

%% TV PARAMETER
tv.rho              = 1e0;
tv.lambda           = 1e-3;
tv.tau              = 1e-1;

tv.nadmm            = 1e2;
tv.nnewton          = 1e1;
tv.nfista           = 1e1;
tv.ntv              = 1e1;

%%
modelPath      	= @(epdir, ep) fullfile(epdir, sprintf('net-epoch-%d.mat', ep));

expDirCTFT    	= [netDir 'cnn_db' num2str(db) '_brn_pretrained_with_ct'];
epochCTFT     	= findLastCheckpoint(expDirCTFT);
load(modelPath(expDirCTFT, epochCTFT));	netCTFT = net;	statsCTFT = stats;

expDirMRFT    	= [netDir 'cnn_db' num2str(db) '_brn_pretrained_with_hcp'];
epochMRFT     	= findLastCheckpoint(expDirMRFT);
load(modelPath(expDirMRFT, epochMRFT));	netMRFT = net;	statsMRFT = stats;

expDirMRONLY   	= [netDir 'cnn_db' num2str(db) '_brn_trained_with_mr_only'];
epochMRONLY   	= findLastCheckpoint(expDirMRONLY);
load(modelPath(expDirMRONLY, epochMRONLY));	netMRONLY = net;	statsMRONLY = stats;

netCTFT.layers(end)     = [];
netMRFT.layers(end)     = [];
netMRONLY.layers(end)   = [];

%%
epochMIN	= min([epochMRFT, epochMRONLY]);

for i = 1:epochMIN
    objMRFT.val(i)      = statsMRFT.val(i).objective;
    psnrMRFT.val(i)     = statsMRFT.val(i).psnr;
    nmseMRFT.val(i)     = statsMRFT.val(i).nrmse;
    
    objMRFT.train(i)	= statsMRFT.train(i).objective;
    
    objMRONLY.val(i)	= statsMRONLY.val(i).objective;
    psnrMRONLY.val(i)	= statsMRONLY.val(i).psnr;
    nmseMRONLY.val(i)	= statsMRONLY.val(i).nrmse;
    
    objMRONLY.train(i)	= statsMRONLY.train(i).objective;
end

%%
dsr                 = 3;

view                = 180;
view_ds             = view/dsr;

wgt                 = 5e3;

dataDir_            = [dataDir 'imdb_brn_subject1.mat'];
imdb_               = load(dataDir_);

opts_IMG.wgt       	= wgt;
opts_IMG.offset     = 0;
opts_IMG.imageSize  = [512, 512, 1];

opts_IMG.inputSize  = [512, 512, 1];
opts_IMG.kernalSize = [0, 0, 1];

opts_IMG.meanNorm   = false;
opts_IMG.varNorm    = false;
opts_IMG.batchSize  = batchSize;
opts_IMG.gpus       = gpus;
opts_IMG.method     = method;
opts_IMG.size       = imageSize;
opts_IMG.set        = 1;

imdb         	= MakeDataMRI(imdb_.images.p, dsr);

view_           = imdb.meta.view_dw;
y               = imdb.images.sino_dw;
theta           = imdb.images.theta_dw;

disp(['...................... [ subject1 ] Reconstruction results from ' num2str(view_) ' views']);
disp(['          .......... Trained with ' num2str(db) ' MR slice']);

%% LABELS
labels                  = imdb.images.labels;

%% DATA
data                    = imdb.images.data;

%% MR-only training
recMRONLY               = recon_cnn4img_simplenn(netMRONLY, data, opts_IMG);

%% CT-pre_training & MR-fine_tuning
recCTFT                 = recon_cnn4img_simplenn(netCTFT, data, opts_IMG);

%% MR-pre_training & MR-fine_tuning
recMRFT                 = recon_cnn4img_simplenn(netMRFT, data, opts_IMG);

%%
labels              = flip(labels, 1);
data                = flip(data, 1);

recMRONLY           = flip(recMRONLY, 1);
recCTFT           	= flip(recCTFT, 1);
recMRFT           	= flip(recMRFT, 1);

maxval              = max(labels(:));

nmse_data           = nmse(max(data, 0)./maxval,labels./maxval);
nmse_recMRONLY      = nmse(recMRONLY./maxval,   labels./maxval);
nmse_recCTFT        = nmse(recCTFT./maxval,     labels./maxval);
nmse_recMRFT        = nmse(recMRFT./maxval,     labels./maxval);

psnr_data           = psnr(max(data, 0)./maxval,labels./maxval);
psnr_recMRONLY      = psnr(recMRONLY./maxval,   labels./maxval);
psnr_recCTFT        = psnr(recCTFT./maxval,     labels./maxval);
psnr_recMRFT        = psnr(recMRFT./maxval,     labels./maxval);



%% FIGURE 8
figure(1);

subplot(1,2,1);
hold on;
plot(objMRONLY.train, 'g:');
plot(objMRFT.train, 'm:');

plot(objMRONLY.val, 'b-');
plot(objMRFT.val, 'r-');
hold off;

legend('[Train] MR-only', '[Train] Proposed (1: HCP)', '[Validation] MR-only', '[Validation] Proposed (1: HCP)');
ylabel('(a) Objective');
xlabel('The number of epoch');
title(['Figure 8(a): Trained with ' num2str(db) ' MR slice']);
grid on;
grid minor;
ylim([0, 1e2]);

subplot(1,2,2);
hold on;
plot(psnrMRONLY.val, 'b-');
plot(psnrMRFT.val, 'r:');
hold off;

legend('[Validation] MR-only', '[Validation] Proposed (1: HCP)');
ylabel('(b) PSNR');
xlabel('The number of epoch');
title(['Figure 8(b): Trained with ' num2str(db) ' MR slice']);
grid on;
grid minor;
ylim([29, 33]);

drawnow();

%% FIGURE 9

wnd                 = [0.00, 0.1];
wnd_df              = [-0.008, +0.008];

% WHOLE IMAGE

bd  = 1:512 - 100;
ix  = 70 + bd;
iy  = 60 + bd;

figure(10); colormap(gray(256));

suptitle('FIgure 9');

subplot(3,4,1); imagesc(labels(iy, ix), wnd);       axis image off; title('Ground truth: Retrospective  (180 views)');

subplot(3,4,5); imagesc(data(iy, ix), wnd);         axis image off; title({'X: Input (60 views)', ['NMSE: ' num2str((nmse_data), '%0.4e')]});
subplot(3,4,6);	imagesc(recMRONLY(iy, ix), wnd);	axis image off; title({'MR-only', ['NMSE: ' num2str((nmse_recMRONLY), '%0.4e')]});
subplot(3,4,7); imagesc(recCTFT(iy, ix), wnd);      axis image off; title({'Proposed (1: CT)', ['NMSE: ' num2str((nmse_recCTFT), '%0.4e')]});
subplot(3,4,8); imagesc(recMRFT(iy, ix), wnd);      axis image off; title({'Proposed (1: HCP)', ['NMSE: ' num2str((nmse_recMRFT), '%0.4e')]});

subplot(3,4,9); imagesc(labels(iy, ix) - data(iy, ix), wnd_df);         axis image off; title({'X: Input (60 views)', ['NMSE: ' num2str((nmse_data), '%0.4e')]});
subplot(3,4,10);imagesc(labels(iy, ix) - recMRONLY(iy, ix), wnd_df);	axis image off; title({'MR-only', ['NMSE: ' num2str((nmse_recMRONLY), '%0.4e')]});
subplot(3,4,11);imagesc(labels(iy, ix) - recCTFT(iy, ix), wnd_df);      axis image off; title({'Proposed (1: CT)', ['NMSE: ' num2str((nmse_recCTFT), '%0.4e')]});
subplot(3,4,12);imagesc(labels(iy, ix) - recMRFT(iy, ix), wnd_df);      axis image off; title({'Proposed (1: HCP)', ['NMSE: ' num2str((nmse_recMRFT), '%0.4e')]});

return ;
