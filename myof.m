clc;
clear;

if ~isdeployed
    addpath('brox_OF'); % Brox 2004 optical flow
end

cd ('Actions');
dnames = dir('*');
dnames = dnames(3:end);

max_flow = 8; % maximum absolute value of flow
scalef = 128/max_flow;

for j=7:9
    cd(dnames(j).name)
    d = dir('*.mp4');
    for p=1:numel(d)
        fn = d(p).name;
        vidReader = VideoReader(fn);
        nof = vidReader.NumberOfFrames;
        im1 = double(read(vidReader,1));
        im1 = imresize(im1,0.2074);
        for i=2:nof
            im2 = double(read(vidReader,i));
            im2 = imresize(im2,0.2074);

            
            flow = mex_OF(im1,im2); % FROM THOMAS BROX 2004

            x = flow(:,:,1); y = flow(:,:,2);


            mag_flow = sqrt(sum(flow.^2,3));
            ort = atan(y./x);
            flow = flow*scalef;  % scale flow
            flow = flow+128;    % center it around 128
            flow(flow<0) = 0;
            flow(flow>255) = 255; % crop the values below 0 and above 255

            mag_flow = mag_flow*scalef; % same for magnitude
            mag_flow = mag_flow+128;
            mag_flow(mag_flow<0) = 0;
            mag_flow(mag_flow>255) = 255;
            
            ort = ort*scalef; % same for orientation
            ort = ort+128;
            ort(ort<0) = 0;
            ort(ort>255) = 255;
            
            im = uint8(cat(3,flow,mag_flow, ort));
            if i==2
                data = im;
            else
                data = cat(4,data,im);
            end
            im1 = im2;
        end
        save(strcat(fn(1:end-4),'_ofmo'), 'data');
        p
    end
    cd ..
    j
end
