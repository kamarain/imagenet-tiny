%MAKE_TINY_EVAL_CLSLOC Makes new test files for evaluation
%
% You may select only certain classes and then this file
% re-generates new files that are used in evaluation.
%
% See also DEMO_EVAL_CLSLOC2.M and DEVKIT_CONF.M

% Run config
make_tiny_eval_clsloc_conf

load(conf.meta_file);

selected_IDs = nan(length(conf.selected_synsets),1);
totIDs = 0;
for classInd = 1:length(synsets)
  if ~isempty(strmatch(synsets(classInd).WNID,conf.selected_synsets))
    totIDs = totIDs+1;
    selected_IDs(totIDs) = synsets(classInd).ILSVRC2014_ID;
    fprintf('Found classes %2d/%2d: %s\n', totIDs,length(conf.selected_synsets),...
            synsets(classInd).words);
  end;
end;

fd_gt = fopen(conf.val_ground_truth_file,'r');
%fd_bl = fopen(blacklist_file,'r');
fd_gt_new = fopen(conf.new_val_ground_truth_file,'w');
fd_bl_new = fopen(conf.new_val_blacklist_file,'w');
fd_vl_new = fopen(conf.new_val_list,'w');

black_list = load(conf.val_blacklist_file);

[gt_ID gt_count] = fscanf(fd_gt,'%d',1);
gt_lineNo = 0;
tot_val = 0;
while gt_count
  gt_lineNo = gt_lineNo+1;
  if sum(gt_ID == selected_IDs)
    % Print validation image number
    fprintf(fd_vl_new,'%d\n',gt_lineNo);
    % Print new ground truth
    fprintf(fd_gt_new,'%d\n',gt_ID);
    tot_val = tot_val+1;
    % Print blacklist if listed in the original
    if sum(gt_lineNo == black_list)
      fprintf(fd_bl_new,'%d\n',tot_val);
    end
    fprintf('\r Tot. validation images: %6d',tot_val);
  end;
  [gt_ID gt_count] = fscanf(fd_gt,'%d',1);
end;

fclose(fd_gt);
%fclose(fd_bl);
fclose(fd_gt_new);
fclose(fd_bl_new);
fclose(fd_vl_new);
fprintf('\n --- All done - Super! --- \n'); 