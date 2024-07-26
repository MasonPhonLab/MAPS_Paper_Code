## only have 21840 aligned files in train set
## only have 3886 aligned files in val set
## only have 6417 aligned files in test set

wdname = "C:/Users/Matt/MAPS_Paper_Code/py_scripts/mfa_boundary_eval_res"
setwd(wdname)

mfa_row = function(set="train") {
  errs = read.csv(file.path(wdname, set, "mfa_all_res.csv"))[,1]
  mdae = round(median(abs(errs)) * 1000, 2)
  mnae = round(mean(abs(errs)) * 1000, 2)
  
  md_str = paste(mdae)
  mn_str = paste(mnae)
  
  cat(paste("MFA", "---", set, mn_str, md_str, sep=" & "), "\\\\")
}

mfa_by_corpus = function(set="train", corpus="timit") {
  all_df = read.csv(file.path(wdname, set, "mfa_all_res.csv"))
  all_df = all_df[all_df$corpus == corpus,]
  errs = all_df$err
  
  mdae = round(median(abs(errs)) * 1000, 2)
  mnae = round(mean(abs(errs)) * 1000, 2)
  
  md_str = paste(mdae)
  mn_str = paste(mnae)
  
  cat(paste("MFA", set, corpus, mn_str, md_str, sep=" & "), "\\\\")
}

mfa_row("train")
mfa_row("val")
mfa_row("test")

mfa_by_corpus(set="test", corpus="timit")
mfa_by_corpus(set="test", corpus="buckeye")

mfa_quantiles = function(set="train") {
  errs = abs(read.csv(file.path(wdname, set, "mfa_all_res.csv"))[,1])
  
  lt10 = round(sum(errs < 0.01) / length(errs) * 100, 2)
  lt20 = round(sum(errs < 0.02) / length(errs) * 100, 2)
  lt25 = round(sum(errs < 0.025) / length(errs) * 100, 2)
  lt50 = round(sum(errs < 0.05) / length(errs) * 100, 2)
  lt100 = round(sum(errs < 0.1) / length(errs) * 100, 2)
  
  s_out = paste("MFA", "---", set, lt10, lt20, lt25, lt50, lt100, sep=" & ")
  cat(s_out, " \\\\\n", sep="")
}

mfa_quantiles(set="train")
mfa_quantiles(set="val")
mfa_quantiles(set="test")

mfa_bisegment_analysis = function(set="test") {
  
  mfa_all_df = read.csv(file.path(set, "mfa_all_res.csv"))
  mfa_all_df = mfa_all_df[mfa_all_df$segment != "<EXCLUDE-name>",]
  mfa_all_df = mfa_all_df[mfa_all_df$next_segment != "<EXCLUDE-name>",]
  mfa_all_df$segment[mfa_all_df$segment == "sil"] = "h#"
  mfa_all_df$next_segment[mfa_all_df$next_segment == "sil"] = "h#"
  
  mfa_all_df$err = mfa_all_df$err * 1000
  mfa_all_df = mfa_all_df[mfa_all_df$next_segment != "#",]
  
  category_df = read.csv("segment_categories.csv")
  
  new_df = merge(mfa_all_df, category_df, by.x="segment", by.y="segment")
  colnames(new_df)[colnames(new_df) == "category"] = "segment_category"
  new_df = merge(new_df, category_df, by.x="next_segment", by.y="segment")
  colnames(new_df)[colnames(new_df) == "category"] = "next_segment_category"
  
  agg = aggregate(err ~ segment_category + next_segment_category,
                  data=new_df, FUN=function(x) median(abs(x)))
  
  agg_sort = agg[order(agg$segment_category),]
  agg_sort$weight = with(new_df, table(paste(segment_category, next_segment_category)) / nrow(new_df))
  sum(agg_sort$err * agg_sort$weight)
  
  outname = "mfa_by_category_means.csv"
  write.csv(agg, file=outname, row.names=FALSE, quote=FALSE)
  system(paste("julia make_heatmap.jl", outname, sep=" "))
  
  make_category_table = function(x_mn) {
    nxt_sgs = sort(unique(x_mn$next_segment_category))
    to_fill = matrix("none", nrow=7, ncol=7)
    cap_sgs = paste(toupper(substr(nxt_sgs, 1, 1)),
                    substring(nxt_sgs, 2),
                    collapse=NULL, sep="")
    d = data.frame(segment=nxt_sgs)
    d = cbind(d, to_fill)
    colnames(d)[2:8] = nxt_sgs
    
    for(s1 in nxt_sgs) {
      for(s2 in nxt_sgs) {
        mn = x_mn$err[x_mn$segment_category == s1 & x_mn$next_segment_category == s2]
        mn = as.character(round(mn, digits=2))
        i1 = which(nxt_sgs == s1)
        i2 = which(nxt_sgs == s2) + 1
        d[i1, i2] = paste("$", mn, "$", sep="")
        if(s1 == "silence" && s2 == "silence") d[i1, i2] = "---"
      }
    }
    
    colnames(d)[1] = "Segment"
    d$Segment = cap_sgs
    colnames(d)[2:8] = cap_sgs
    return(d)
  }
  tb = make_category_table(agg)
  cat(paste(colnames(tb), collapse=" & ")); cat(" \\\\\n")
  for(i in 1:nrow(tb)) {
    r = tb[i,]
    s = paste(r, collapse=" & ", sep="")
    s = paste(s, " \\\\\n", sep="")
    cat(s)
  }
}

mfa_bisegment_analysis(set="test")
