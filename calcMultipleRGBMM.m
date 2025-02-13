clc
clear;

cd Actions

dnames = dir('*');
dnames = dnames(3:end);
ctr_frames = 0;

for i=1:numel(dnames)
    cd(dnames(i).name)
    fnames = dir('*ofmo.mat');
    for k=1:numel(fnames)
        fn = fnames(k).name;
        load(fn);
        nof = size(data,4);
        L = floor(nof/4);
        startFrame = 1;
        endFrame = startFrame + L-1;
        ctr = 1;
        while endFrame <= nof
            im1 = data(:,:,1:3,startFrame);
            mhi = 0;
            for p=startFrame+1:endFrame
                im2 = data(:,:,1:3,p);
                mhi = mhi+ double(abs(im2-im1));
                im1 = im2;
            end
            mhi = mhi/max(mhi(:));
            imwrite(mhi, strcat('../../ofi4/', fn(1:end-4), '_', int2str(ctr), '_ofi.jpg'));
            ctr = ctr+1;
            startFrame = endFrame;
            endFrame = startFrame + L-1;
            if ctr == 5
                break;
            end
        end
    end
    cd ..;
    i
end



