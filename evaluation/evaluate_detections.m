function [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections, count] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path, draw, target_class, IOU)
% 'bboxes' is Nx4, N is the number of non-overlapping detections, and each
% row is [x_min, y_min, x_max, y_max]
% 'confidences' is the Nx1 (final cascade node) confidence of each
% detection.
% 'image_ids' is the Nx1 image names for each detection. 

%This code is modified from the 2010 Pascal VOC toolkit.
%http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/index.html#devkit

if(~exist('draw', 'var'))
    draw = 1;
end

%this lists the ground truth bounding boxes for the test set.
fid = fopen(label_path);

% fmt = '(%d, %d) - (%d, %d) - (%s)';
% gt_info = textscan(fid,fmt,'Delimiter','()','MultipleDelimsAsOne',true);

count = 1;
while ~feof(fid)
   tline = fgetl(fid);
   pred_info = strsplit(tline);
   
   %class
   tmp5=strsplit(pred_info{7},'(');tmp52=strsplit(tmp5{2},')');
   if strcmp(tmp52{1},target_class) ~= 1
       continue
   end
   gt_info{count,5} = tmp52{1};

   
   % xmin
   tmp1=strsplit(pred_info{1},'(');tmp12=strsplit(tmp1{2},',');
   gt_info{count,1} = tmp12{1};
   
   %ymin
   tmp2=strsplit(pred_info{2},')');
   gt_info{count,2} = tmp2{1};
   
   %xmax
   tmp3=strsplit(pred_info{4},'(');tmp32=strsplit(tmp3{2},',');
   gt_info{count,3} = tmp32{1};
   
   %ymax
   tmp4=strsplit(pred_info{5},')');
   gt_info{count,4} = tmp4{1};
   
   count = count +1;
end

count =  count -1;

fclose(fid);

% assign detections to ground truth objects
nd=length(confidences);
tp=zeros(nd,1);
fp=zeros(nd,1);
duplicate_detections = zeros(nd,1);
try
    gt_ids = gt_info{1,5}; % string
catch
    gt_ids=[]; gt_bboxes=[]; gt_isclaimed=[]; fp=ones(nd,1); duplicate_detections=[];
    return
end    

gt_bboxes = [str2double(gt_info{1,1}), str2double(gt_info{1,2}), str2double(gt_info{1,3}), str2double(gt_info{1,4})];
gt_bboxes = double(gt_bboxes);

gt_isclaimed = zeros(length(gt_ids),1);
npos = size(gt_ids,1); %total number of true positives.

% sort detections by decreasing confidence
% c_idx = find(confidences>0.6);
% confidences = confidences(c_idx);
% image_ids = image_ids(c_idx);
% bboxes = bboxes(c_idx);
[sc,si]=sort(-confidences);
image_ids= image_ids(si);
bboxes = bboxes(si,:);

% assign detections to ground truth objects, move ahead
%nd=length(confidences);
%tp=zeros(nd,1);
%fp=zeros(nd,1);
%duplicate_detections = zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('pr: compute: %d/%d\n',d,nd);
        drawnow;
        tic;
    end
%     cur_gt_ids = strcmp(image_ids{d}, gt_ids); %will this be slow?
    cur_gt_ids = strcmp(target_class, gt_ids); %change it to multi-classes

    bb = bboxes(d,:);
    ovmax=-inf;

    for j = find(cur_gt_ids')
        bbgt=gt_bboxes(j,:);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0       
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih; % iw*ih => intersection
            ov=iw*ih/ua;
            if ov>ovmax %higher overlap than the previous best?
                ovmax=ov;
                jmax=j;
            end
        end
    end
    
    % assign detection as true positive/don't care/false positive
    if ovmax >= IOU % 0.3
        if ~gt_isclaimed(jmax)
            tp(d)=1;            % true positive
            gt_isclaimed(jmax)=true;
        else
            fp(d)=1;            % false positive (multiple detection)
            duplicate_detections(d) = 1;
        end
    else
        fp(d)=1;                    % false positive
    end
end

% % compute cumulative precision/recall
% cum_fp=cumsum(fp);
% cum_tp=cumsum(tp);
% rec=cum_tp/npos; %npo : number of ground truth positive
% prec=cum_tp./(cum_fp+cum_tp); % precision
% 
% ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    figure(12)
    plot(rec,prec,'-');
    axis([0 1 0 1])
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('Average Precision = %.3f',ap));
    set(12, 'Color', [.988, .988, .988])
    
    pause(0.1) %let's ui rendering catch up
    average_precision_image = frame2im(getframe(12));
    % getframe() is unreliable. Depending on the rendering settings, it will
    % grab foreground windows instead of the figure in question. It could also
    % return a partial image.
    imwrite(average_precision_image, 'visualizations/average_precision.png')
    
    figure(13)
    plot(cum_fp,rec,'-')
    axis([0 300 0 1])
    grid;
    xlabel 'False positives'
    ylabel 'Number of correct detections (recall)'
    title('This plot is meant to match Figure 6 in Viola Jones');
end

%% Re-sort return variables so that they are in the order of the input bboxes
reverse_map(si) = 1:nd;
tp = tp(reverse_map);
fp = fp(reverse_map);
duplicate_detections = duplicate_detections(reverse_map);



