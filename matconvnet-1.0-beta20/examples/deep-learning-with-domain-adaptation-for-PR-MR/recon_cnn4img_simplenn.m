function rec = recon_cnn4img_simplenn(net, data, opts)

if ~isempty(opts.gpus)
%     net.move('gpu');
    net = vl_simplenn_move(net, 'gpu');
end

rec     = zeros(opts.size, 'single');
data    = opts.wgt*data + opts.offset;
set     = opts.set;


for ival	= 1:1:length(set)
    
    iz          = set(ival);
    
    data_       = single(squeeze(data(:,:,iz)));
    data_patch	= getBatchPatchVal(data_, opts);
    
    %%
    nbatch      = size(data_patch, 4);
    batch_      = (1:opts.batchSize) - 1;
    
    rec_batch = single([]);
    
    for ibatch  = 1:opts.batchSize:nbatch
        batch                       = ibatch + batch_;
        batch(batch > nbatch)       = [];
        
        data_batch                  = data_patch(:,:,:,batch);
        
        if ~isempty(opts.gpus)
            data_batch	= gpuArray(data_batch);
        end

%         net.forward({'input',data_batch}) ;
%         rec_batch_  	= net.vars(opts.vid).value;
        rec_batch_  = vl_simplenn(net, data_batch, [], [], 'conserveMemory', 0, 'mode', 'test');
        rec_batch_	= rec_batch_(end).x;
        
        if strcmp(opts.method, 'residual')
            rec_batch_ 	= data_batch - rec_batch_;
        end

        rec_batch(:,:,:,batch)      = gather(rec_batch_);
    end
    
    rec(:,:,ival)   = getReconPatchVal(rec_batch, opts);
    
end

rec             = (rec - opts.offset)./opts.wgt;
rec(rec < 0)    = 0;

vl_simplenn_move(net, 'cpu');
net = [];

end