%this script demos the usage of evaluation routines
% the result file 'demo.val.pred.txt' on validation data is evaluated
% against the ground truth

% Run the config setting
devkit_conf

if isempty(conf.predictionFile)
  predictionFile = fullfile(conf.devkitDir,'evaluation/demo.val.pred.loc.txt');
  fprintf(['No prediction file given, using the one that comes with ' ...
           'the development KIT: ' predictionFile '\n']);
else
  predictionFile = conf.predictionFile;
end;

% add the ILSVRC functionality to the path
addpath(fullfile(conf.devkitDir,'evaluation'));

fprintf('CLASSIFICATION WITH LOCALIZATION TASK\n');

meta_file = fullfile(conf.devkitDir,'data/meta_clsloc.mat');
pred_file = predictionFile;
ground_truth_file = fullfile(conf.devkitDir,...
                             'data/ILSVRC2014_clsloc_validation_ground_truth.txt');
blacklist_file = fullfile(conf.devkitDir,...
                          'data/ILSVRC2014_clsloc_validation_blacklist.txt');
ground_truth_dir = conf.ilsvrc_bbox_val_dir;
num_predictions_per_image=5;
optional_cache_file = '';

fprintf('pred_file: %s\n', pred_file);
fprintf('ground_truth_file: %s\n', ground_truth_file);
fprintf('blacklist_file: %s\n', blacklist_file);

if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

num_val_files = -1;
if ~exist('ground_truth_dir','var')
  ground_truth_dir=input('Please enter the path to the Validation bounding box annotations directory: ', 's');
  fprintf('ground_truth_dir: %s\n', ground_truth_dir);
end;

val_files = dir(sprintf('%s/*.xml',ground_truth_dir));
num_val_files = numel(val_files);

if num_val_files == -1 
  error('No XML files found in the given ground_truth_dir!');
elseif num_val_files ~= 50000
  error(['In the given ground_truth_dir there is expected to be ' ...
         '50,000 validation images as defined in ILSVRC2012!']);
end

error_cls = zeros(num_predictions_per_image,1);
error_loc = zeros(num_predictions_per_image,1);

for i=1:num_predictions_per_image
%    [error_cls(i) error_loc(i)] = eval_clsloc(pred_file,ground_truth_file,ground_truth_dir,...
%                                              meta_file,i, blacklist_file,optional_cache_file);
    [error_cls(i)] = eval_clsloc(pred_file,...
                                 ground_truth_file,...
                                 ground_truth_dir,...
                                 meta_file,i,...
                                 blacklist_file,...
                                 optional_cache_file);
end

disp('# guesses vs clsloc error vs cls-only error');
disp([(1:num_predictions_per_image)',error_loc,error_cls]);


