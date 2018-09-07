%% Step 1. Download the trained network
% MR-only network
network_path    = './network/cnn_db1_brn_trained_with_mr_only/';
network_name	= [network_path 'net-epoch-500.mat'];
network_url     = 'https://www.dropbox.com/s/69jdzou1owm9opv/net-epoch-500.mat?dl=1';

mkdir(network_path);
fprintf('downloading MR-only network from %s\n', network_url) ;
websave(network_name, network_url);

% CT pre-trained network
network_path    = './network/cnn_db1_brn_pretrained_with_ct/';
network_name	= [network_path 'net-epoch-500.mat'];
network_url     = 'https://www.dropbox.com/s/6oh5i2b4i2fi8kv/net-epoch-500.mat?dl=1';

mkdir(network_path);
fprintf('downloading CT pre-trained network from %s\n', network_url) ;
websave(network_name, network_url);

% HCP pre-trained network
network_path    = './network/cnn_db1_brn_pretrained_with_hcp/';
network_name	= [network_path 'net-epoch-500.mat'];
network_url     = 'https://www.dropbox.com/s/r9zk34grdg7ob2k/net-epoch-500.mat?dl=1';

mkdir(network_path);
fprintf('downloading HCP pre-trained network from %s\n', network_url) ;
websave(network_name, network_url);
