function [ DownSino Mask] = radial_downsampling( orig , downsampling )
%RADIAL_DOWNSAMPLING Summary of this function goes here
%   Detailed explanation goes here

[nR nA nframe] = size(orig);

DownSino    = zeros(nR,ceil(nA/downsampling),nframe);
Mask        = zeros(ceil(nA/downsampling),nframe);

for iframe = 1: nframe
%     shift = mod(iframe-1, downsampling);
    shift   = 0;
    DownSino(:,:,iframe) = orig(:,1+shift:downsampling:end,iframe);
    Mask(:,iframe) = ([1+shift:downsampling:nA]-1)/nA*180 + 90;
end


end

