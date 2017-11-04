function run_all(result_type, save_name, result_name)
    target_classes = {'chair','table','sofa','bed','toilet'};
    all_ap = 0;
    for kkk = 1:5
        target_class = target_classes{kkk};
        % test
        %fpath = sprintf('../../results/test/comp4-27463_det_test_%s.txt',target_class);
        % train
        filename = sprintf('%s_%s',target_class, result_name);
        IOU = 0.5;
        type = '';
        fpath = sprintf('../result/%s/%s/%s/%s.txt', result_type, save_name, result_name, filename);
        label_parent_dir = sprintf('../data/DIRE/Annotations');
        gt_file_list = dir(label_parent_dir);
        
        [bboxes, confidences, image_ids] = fetch_result(fpath);
    %     save('rcnn_result.mat','bboxes','confidences','image_ids')
        
    %     %% trim low confidence
    %    target_id = find(confidences>0.6);
    %    bboxes = bboxes(target_id);
    %    confidences = confidences(target_id);
    %    image_ids = image_ids(target_id);
        %%
        unique_image = unique(image_ids);

        all_tp=[]; all_fp=[]; all_box_num = 0; all_gt_box_num =0;
        for i = 1:length(unique_image)
            ids = find(image_ids==unique_image(i));
            label_path = fullfile(label_parent_dir,sprintf('picture_%06d.txt',unique_image(i)));
            [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections, obj_count] = ...
                evaluate_detections(bboxes(ids,:), confidences(ids,:), image_ids(ids,:), ...
                label_path, 0, target_class, IOU); % 0: don't draw now
            all_tp = [all_tp;tp];
            all_fp = [all_fp;fp];
            all_box_num = all_box_num + length(tp);
            all_gt_box_num = all_gt_box_num + obj_count;
            
    %         con_idx=find(confidences(ids,:)>0.2);
    %         if ~isempty(tp)
    %             all_tp = [all_tp;tp(con_idx)];
    %             all_fp = [all_fp;fp(con_idx)];
    %             all_box_num = all_box_num + length(tp(con_idx));
    %             all_gt_box_num = all_gt_box_num + obj_count;
    %             confidences = confidences(con_idx);
    %         end
        end
        
        %con_idx=find(confidences>0.6);
        %all_tp = all_tp(con_idx);
        %all_fp = all_fp(con_idx);
        %confidences = confidences(con_idx);
     
        [prec, rec, ap, cum_tp, cum_fp]=compute_cu_pr(all_tp,all_fp,all_gt_box_num,confidences);
        draw(prec, rec, ap, cum_fp, filename,IOU,target_class, result_type, save_name, result_name, filename)
        all_ap = all_ap+ap;
        disp(target_class)
        precision=sum(all_tp)/all_box_num;
        disp(sprintf('precision: %d/%d = %.01f%% \n',sum(all_tp),all_box_num,precision*100));
        recall=sum(all_tp)/all_gt_box_num;
        disp(sprintf('recall: %d/%d = %.01f%% \n',sum(all_tp),all_gt_box_num,recall*100));
        disp(sprintf('AP: %f', ap*100));
        result_path = sprintf('../result/%s/%s/%s/result.txt', result_type, save_name, result_name);
        fileID = fopen(result_path, 'a');
        fprintf(fileID, '%s\n', target_class);
        fprintf(fileID, 'precision: %d/%d = %.01f%% \n',sum(all_tp),all_box_num,precision*100);
        fprintf(fileID, 'recall: %d/%d = %.01f%% \n',sum(all_tp),all_gt_box_num,recall*100);
        fprintf(fileID, 'AP: %f\n\n', ap*100);
        
        

    end

    mAP = all_ap/length(target_classes);
    fprintf(fileID, 'mAP: %f\n', mAP);
    fclose(fileID);
    disp(sprintf('mAP: %f\n', mAP))
