%READ_IMAGENET_TINY
%
% Reads the generated tiny images as a whole to the memory into
% a datastructure. Takes some time and will be a huge blob in the
% memory. Save the data as a Matlab blob (.mat) and don't read
% every time again!
%
% NOTE: you can control the number of images to read in the config
% file
%
% Author: joni.kamarainen at tut.fi
%
% See also READ_IMAGENET_TINY_CONF.M

% Run the config setting
read_ImageNet_tiny_conf

% Load synsets
load(fullfile(conf.imagenetDir,...
              'ILSVRC2014_devkit',...
              'data/meta_clsloc.mat'));

fprintf('Configuration: \n');
conf

% Read training images
fprintf('Reading the tiny train images:\n');
trainTot = 0;
for classNum = 1:1000
  trainDir = fullfile(conf.tinyDir,...
                      conf.imagenetTrainDir,...
                      synsets(classNum).WNID);
  if ~exist(trainDir,'dir')
    error('Tiny Images not Found!');
  end

  % Store example image path
  exampleListing = dir(fullfile(trainDir,'*.JPEG'));
  train_example(classNum).imgFile = fullfile(trainDir,exampleListing(1).name);

  % Read train images
  trainDirListing = dir(fullfile(trainDir,'*_tiny.bmp'));
  if (conf.howMany == -1)
    totRead = length(trainDirListing);
  else
    totRead = conf.howMany;
  end;
  for dirEntry = 1:totRead
    fprintf('\r Class %d/1000: direntry %d/%d    ',classNum,dirEntry, ...
            length(trainDirListing));
    
    tinyImgFile = trainDirListing(dirEntry).name;

    trainTot = trainTot+1;
    train_data(trainTot).img = imread(fullfile(trainDir,tinyImgFile));
    train_class(trainTot) = classNum;
    train_wnid(trainTot,:) = synsets(classNum).WNID;
  end;
end;
fprintf(' Done!\n');

if (conf.readVal)
  % Convert validation images
  fprintf('Read also the validation images:\n');
  valDir = fullfile(conf.tinyDir,...
                    conf.imagenetValDir);
  valDirListing = dir(fullfile(valDir,'*_tiny.bmp'));
  if (isempty(valDirListing))
    error([valDir ' no *_tiny.bmp test images - cannot read validation set!']);
  end;
  gt = load(fullfile(conf.imagenetDir,...
                     'ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt'));
  
  if (conf.howManyVal == -1)
    totRead = 50000;
  else
    totRead = conf.howManyVal;
  end;

  for dirEntry = 1:totRead
    fprintf('\r Direntry %d/%d  ',dirEntry, totRead);
    tinyImgFile = sprintf('ILSVRC2012_val_%08d_tiny.bmp',dirEntry);
    
    val_data(dirEntry).img = imread(fullfile(valDir,tinyImgFile));
    val_class(dirEntry) = gt(dirEntry);
    val_wnid(dirEntry,:) = synsets(gt(dirEntry)).WNID;
  end;
fprintf(' Done!\n');
end;

clear classNum dirEntry exampleListing tinyImgFile totRead trainDir ...
    trainDirListing trainTot valDir valDirListing