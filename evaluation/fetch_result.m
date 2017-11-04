function [bboxes, confidences, image_ids] = fetch_result(fpath)
% fpath='../../results/train/test.txt'
% label_path = '../../data/DIRE/Annotations/picture_000003_01.txt'


fid = fopen(fpath);
% picture_000003 0.999 86.3 1.0 204.0 18.3
count = 1;
while ~feof(fid)
   tline = fgetl(fid);
   pred_info = strsplit(tline);
   bboxes(count,1:4) = [str2double(pred_info{3}), str2double(pred_info{4}),...
       str2double(pred_info{5}), str2double(pred_info{6})];
   confidences(count,1) = str2double(pred_info{2});
   C = strsplit(strrep(pred_info{1},'picture_',''),'_');
   image_ids(count,1) = str2double(C{1});
   count = count +1;
end

