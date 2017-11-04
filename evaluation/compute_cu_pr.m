function [prec, rec, ap, cum_tp, cum_fp]=compute_cu_pr(tp,fp,npos,confidences)

% sorting according to confidence
[sc,si]=sort(-confidences);
tp= tp(si);
fp= fp(si);

% compute cumulative precision/recall
cum_fp=cumsum(fp);
cum_tp=cumsum(tp);
rec=cum_tp/npos; %npo : number of ground truth positive
prec=cum_tp./(cum_fp+cum_tp); % precision

ap=VOCap(rec,prec);