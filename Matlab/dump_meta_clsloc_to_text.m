% Run the config setting
devkit_conf

% add the ILSVRC functionality to the path
addpath(fullfile(conf.devkitDir,'evaluation'));

fprintf('CLASSIFICATION WITH LOCALIZATION TASK\n');

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

fprintf('Dumped %d lines to %s\n',tot_classes,dump_fname); 
