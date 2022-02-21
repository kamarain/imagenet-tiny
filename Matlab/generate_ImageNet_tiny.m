%GENERATE_IMAGENET_TINY
%
% Generates a "tiny" version of the ImageNet dataset
% that is smaller images for nice little experiments on
% machine learning and computer vision
%
% Author: joni.kamarainen at tut.fi
%
% See also GENERATE_IMAGENET_TINY_CONF.M

% Run the config setting
generate_ImageNet_tiny_conf

% Load synsets
%load(fullfile(conf.imagenetDir,...
%              'ILSVRC2014_devkit',...
%              'data/meta_clsloc.mat'));
load(fullfile(conf.devkitDir,...
              'data/meta_clsloc.mat'));


fprintf('Configuration: \n');
conf

% Make a proper conversion file for the ImageMagick convert command
% line tool
switch conf.convType
 case 1,
  % Keep aspect ratio and center image
  convString =[' -resize ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
              ' -background black -gravity center'...
              ' -extent ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
              ' -colorspace ' conf.colorSpace ...
              ' -depth ' num2str(conf.bitDepth) ' '] 
 case 2,
  % Ignore aspect ratio and center image
  convString =[' -resize ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
               '\!'...
              ' -background black -gravity center'...
              ' -extent ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
              ' -colorspace ' conf.colorSpace ...
              ' -depth ' num2str(conf.bitDepth) ' '] 
 case 3,
  % Keep aspect ratio, center and crop central part
  convString =[' -resize ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
               '^'...
              ' -background black -gravity center'...
              ' -crop ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) '+0+0'...
              ' -colorspace ' conf.colorSpace ...
              ' -depth ' num2str(conf.bitDepth) ' '] 

 otherwise,
  error(['Unknown coversions type conf.convType = ' num2str(conf.convType)]);
end;
  
% Convert training images
fprintf('Converting the training images:\n');
for classNum = 1:1000
  srcDir = fullfile(conf.imagenetDir,...
                    conf.imagenetTrainDir,...
                    synsets(classNum).WNID);
  srcDirListing = dir(srcDir);
  if (isempty(srcDirListing))
    warning([srcDir ' does not exist!']);
    continue
  end;
  firstImgFound = false;
  trgDir = fullfile(conf.tinyDir,...
                    conf.imagenetTrainDir,...
                    synsets(classNum).WNID);
  if ~exist(trgDir,'dir')
    mkdir(trgDir);
  elseif ~conf.enforce
    warning([trgDir ' exists - skipping!']);
    continue;
  end
  for dirEntry = 1:length(srcDirListing)
    fprintf('\r Class %d/1000: direntry %d/%d',classNum,dirEntry, ...
            length(srcDirListing));
    
    % Convert a tiny version of the original image
    if length(srcDirListing(dirEntry).name) > 4 && ...
          strmatch(srcDirListing(dirEntry).name(end-3:end),'JPEG')
      %Template: convert src.JPEG -resize 256x256 -background black\
      % -gravity center -extent 256x256 -colorspace Gray -depth 8 trg.bmp 
      system(['convert '...
              fullfile(srcDir,srcDirListing(dirEntry).name) ...
              convString,...
              fullfile(trgDir,...
                       [srcDirListing(dirEntry).name(1:end-5) '_tiny.bmp'])]);
    
      % Copy the first image just as an example of the originals
      if (~firstImgFound && conf.debugLevel > 0) || conf.debugLevel > 1
        firstImgFound = true;
        firstImgFile = srcDirListing(dirEntry).name;
        tinyImgFile = [srcDirListing(dirEntry).name(1:end-5) '_tiny.bmp'];
        system(['cp ' fullfile(srcDir,firstImgFile) ' ' trgDir]);

        % Draw also the first image and its class label
        firstImg = imread(fullfile(srcDir,firstImgFile));
        tinyImg = imread(fullfile(trgDir,tinyImgFile));
        subplot(1,2,1);
        imshow(firstImg);
        title([synsets(classNum).WNID ':' synsets(classNum).words],'FontSize',20);
        subplot(1,2,2);
        imshow(tinyImg);
        title('tiny version','FontSize',20);
        drawnow
        if conf.debugLevel > 1
          input('DEBUG[2] - press return');
        end;
      end;
    end;
  end;
end;
fprintf(' Done!\n');

% Convert validation images
fprintf('Converting the validation images:\n');
srcDir = fullfile(conf.imagenetDir,...
                  conf.imagenetValDir);
srcDirListing = dir(srcDir);
if (isempty(srcDirListing))
  error([srcDir ' does not exist - cannot do the validation set!']);
  %continue
end;
trgDir = fullfile(conf.tinyDir,...
                  conf.imagenetValDir);
if ~exist(trgDir,'dir')
  mkdir(trgDir);
  startInd = 1;
else
  trgDirListing = dir(trgDir);
  startInd = max([length(trgDirListing)-5 1]);
end
if conf.enforce
  startInd = 1;
else
  fprintf('Re-using already generated and starting from entry %d',...
          startInd);
end;
for dirEntry = startInd:length(srcDirListing)
  fprintf('\r Direntry %d/%d    ',dirEntry, length(srcDirListing));
    
  % Convert a tiny version of the original image
  if length(srcDirListing(dirEntry).name) > 4 && ...
        strmatch(srcDirListing(dirEntry).name(end-3:end),'JPEG')
    %Template: convert src.JPEG -resize 256x256 -background black\
    % -gravity center -extent 256x256 -colorspace Gray -depth 8 trg.bmp 
    system(['convert '...
            fullfile(srcDir,srcDirListing(dirEntry).name) ...
            ' -resize ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
            ' -background black -gravity center'...
            ' -extent ' num2str(conf.tiny_width) 'x' num2str(conf.tiny_height) ...
            ' -colorspace ' conf.colorSpace ...
            ' -depth 8 '...
            fullfile(trgDir,...
                     [srcDirListing(dirEntry).name(1:end-5) '_tiny.bmp'])]);
    
  end;
end;
fprintf(' Done!\n');
