% Run the config setting
devkit_conf

% add the ILSVRC functionality to the path
addpath(fullfile(conf.devkitDir,'evaluation'));

fprintf('CLASSIFICATION WITH LOCALIZATION TASK\n');
fprintf(' => Dumping training data information...\n');

meta_file = fullfile(conf.devkitDir,'data/meta_clsloc.mat');

load(meta_file);

dump_fname = 'DUMP_meta_clsloc.txt';

fid = fopen(dump_fname,'w');

tot_classes = 0;
for i = 1:length(synsets)
  if synsets(i).num_train_images > 0
    tot_classes = tot_classes+1;
    fprintf(fid, '{%d} {%s} {%s} {%d}\n',...
	    synsets(i).ILSVRC2014_ID,...
	    synsets(i).WNID,...
	    synsets(i).words,...
	    synsets(i).num_train_images);
  end;
end;

fclose(fid);

fprintf(' => Done - Dumped %d lines to %s\n',tot_classes,dump_fname); 

fprintf(' => Dumping validation data information...\n');

blacklisted_file = fullfile(conf.devkitDir,'data/ILSVRC2014_clsloc_validation_blacklist.txt');
ground_truth_file = fullfile(conf.devkitDir,'data/ILSVRC2014_clsloc_validation_ground_truth.txt');

blacklisted = load(blacklisted_file);
ground_truth = load(ground_truth_file);

dump_fname_val = 'DUMP_validation.txt';
fid2 = fopen(dump_fname_val,'w');

tot_val = 0;
for val_num = 1:50000
  if sum(blacklisted == val_num) == 0
    tot_val = tot_val + 1;
    gt = ground_truth(val_num);
    img_file_prefix = sprintf('ILSVRC2012_val_%08d',val_num);
    fprintf(fid2, '{%s} {%d}\n',...
	    img_file_prefix,...
	    gt);
  end;
end;

fclose(fid);

fprintf(' => Done - Dumped %d lines to %s\n',tot_val,dump_fname_val); 
