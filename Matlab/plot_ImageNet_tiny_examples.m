%PLOT_IMAGENET_TINY_EXAMPLES
%
% Plots the first image of each class along with its
% "tiny" version. Utilises the generation script config.
%
% Author: joni.kamarainen at tut.fi
%
% See also GENERATE_IMAGENET_TINY_CONF.M

% Run the config setting
generate_ImageNet_tiny_conf

% Load synsets
load(fullfile(conf.imagenetDir,...
              'ILSVRC2014_devkit',...
              'data/meta_clsloc.mat'));

fprintf('Configuration: \n');
conf

% Convert training images
fprintf('Converting the training images:\n');
for classNum = 1:1000
  srcDir = fullfile(conf.tinyDir,...
                    conf.imagenetTrainDir,...
                    synsets(classNum).WNID);
  srcDirListing = dir(srcDir);
  if (isempty(srcDirListing))
    warning([srcDir ' does not exist!']);
    continue
  end;
  firstImgFound = false;
  for dirEntry = 1:length(srcDirListing)
    fprintf('\r Class %d/1000: direntry %d/%d',classNum,dirEntry, ...
            length(srcDirListing));
    
    % Convert a tiny version of the original image
    if length(srcDirListing(dirEntry).name) > 4 && ...
          strmatch(srcDirListing(dirEntry).name(end-3:end),'JPEG')
      tinyImgFile = [srcDirListing(dirEntry).name(1:end-5) '_tiny.bmp'];
      origImgFile = srcDirListing(dirEntry).name;

      % Draw also the first image and its class label
      origImg = imread(fullfile(srcDir,origImgFile));
      tinyImg = imread(fullfile(srcDir,tinyImgFile));
      subplot(1,2,1);
      imshow(origImg);
      title([synsets(classNum).WNID ':' synsets(classNum).words],'FontSize',20);
      subplot(1,2,2);
      imshow(tinyImg);
      title('tiny version','FontSize',20);
      input('First image and its tiny version <RETURN>');
      break;
    end;
  end;
end;
fprintf(' Done!\n');
