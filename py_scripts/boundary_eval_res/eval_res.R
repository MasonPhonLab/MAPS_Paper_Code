## total files in train set:  45855
## total files in val set:    3890
## total files in test set:   6421

wdname = "C:/Users/Matt/MAPS_Paper_Code/py_scripts/boundary_eval_res"
setwd(wdname)
all_sparse_filenames = list.files("val", pattern="full_real.*all_res")
all_crisp_filenames = list.files("val", pattern="^real_seed.*all_res")

crisp_row = function(set="train", style="interp") {
  all_df = lapply(all_crisp_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  all_df = Reduce(rbind, all_df)
  mdae = tapply(all_df[,style], all_df$round, function(x) median(abs(x)))
  mnae = tapply(all_df[,style], all_df$round, function(x) mean(abs(x)))
  
  md_sterr = sd(mdae) / sqrt(length(mdae)) * 1000
  mn_sterr = sd(mnae) / sqrt(length(mnae)) * 1000
  
  md_str = paste("$", round(mean(mdae) * 1000, 2), " \\pm ", round(1.96 * md_sterr, 2), "$", sep="")
  mn_str = paste("$", round(mean(mnae) * 1000, 2), " \\pm ", round(1.96 * mn_sterr, 2), "$", sep="")
  
  cat(paste("Crisp", ifelse(style == "interp", "Yes", "No"), set, mn_str, md_str, sep=" & "), "\\\\\n")
}

crisp_by_corpus = function(set="train", style="interp", corpus="timit") {
  all_df = lapply(all_crisp_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  all_df = Reduce(rbind, all_df)
  all_df = all_df[all_df$corpus == corpus,]
  mdae = tapply(all_df[,style], all_df$round, function(x) median(abs(x)))
  mnae = tapply(all_df[,style], all_df$round, function(x) mean(abs(x)))
  
  md_sterr = sd(mdae) / sqrt(length(mdae)) * 1000
  mn_sterr = sd(mnae) / sqrt(length(mnae)) * 1000
  
  md_str = paste("$", round(mean(mdae) * 1000, 2), " \\pm ", round(1.96 * md_sterr, 2), "$", sep="")
  mn_str = paste("$", round(mean(mnae) * 1000, 2), " \\pm ", round(1.96 * mn_sterr, 2), "$", sep="")
  
  cat(paste("Crisp", ifelse(style == "interp", "Yes", "No"), set, corpus, mn_str, md_str, sep=" & "), "\\\\\n")
}

sparse_row = function(set="train", style="interp") {
  all_df = lapply(all_sparse_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  all_df = Reduce(rbind, all_df)
  mdae = tapply(all_df[,style], all_df$round, function(x) median(abs(x)))
  mnae = tapply(all_df[,style], all_df$round, function(x) mean(abs(x)))
  
  md_sterr = sd(mdae) / sqrt(length(mdae)) * 1000
  mn_sterr = sd(mnae) / sqrt(length(mnae)) * 1000
  
  md_str = paste("$", round(mean(mdae) * 1000, 2), " \\pm ", round(1.96 * md_sterr, 2), "$", sep="")
  mn_str = paste("$", round(mean(mnae) * 1000, 2), " \\pm ", round(1.96 * mn_sterr, 2), "$", sep="")
  
  cat(paste("Sparse", ifelse(style == "interp", "Yes", "No"), set, mn_str, md_str, sep=" & "), "\\\\\n")
}

crisp_row(set="train", style="interp")
crisp_row(set="val", style="interp")
crisp_row(set="test", style="interp")
crisp_row(set="train", style="nointerp")
crisp_row(set="val", style="nointerp")
crisp_row(set="test", style="nointerp")

sparse_row(set="train", style="interp")
sparse_row(set="val", style="interp")
sparse_row(set="test", style="interp")
sparse_row(set="train", style="nointerp")
sparse_row(set="val", style="nointerp")
sparse_row(set="test", style="nointerp")

crisp_by_corpus(set="test", style="interp", corpus="timit")
crisp_by_corpus(set="test", style="interp", corpus="buckeye")

crisp_quantiles = function(set="train", style="interp") {
  all_df = lapply(all_crisp_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  all_df = Reduce(rbind, all_df)
  lt10 = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < 0.01) / length(x)) |> mean() * 100
  lt20 = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < 0.02) / length(x)) |> mean() * 100
  lt25 = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < 0.025) / length(x)) |> mean() * 100
  lt50 = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < 0.05) / length(x)) |> mean() * 100
  lt100 = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < 0.1) / length(x)) |> mean() * 100
  
  s = cat("Crisp", ifelse(style=="interp", "Yes", "No"), set, round(lt10, 2), round(lt20, 2), round(lt25, 2), round(lt50, 2), round(lt100, 2), sep=" & ")
  cat(s, "\\\\\n")
}

crisp_quantiles_w_se = function(set="train", style="interp") {
  all_df = lapply(all_crisp_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  cutoffs = c(0.01, 0.02, 0.025, 0.05, 0.1)
  all_df = Reduce(rbind, all_df)
  s_total = paste("Crisp &", ifelse(style=="interp", "Yes", "No"), "&", set)
  for(cut in cutoffs) {
    res_vec = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < cut) / length(x))
    mn = round(mean(res_vec) * 100, 2)
    se = round(sd(res_vec) / sqrt(length(res_vec)) * 1.96 * 100, 2)
    s_part = paste(mn, paste("$\\pm ", se, "$", sep=""), sep="")
    s_total = paste(s_total, s_part, sep=" & ")
  }
  
  cat(s_total, "\\\\\n")
}

sparse_quantiles = function(set="train", style="interp") {
  all_df = lapply(all_sparse_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  all_df = Reduce(rbind, all_df)
  lt10 = round(sum(abs(all_df[,style]) < 0.01) / nrow(all_df) * 100, 2)
  lt20 = round(sum(abs(all_df[,style]) < 0.02) / nrow(all_df) * 100, 2)
  lt25 = round(sum(abs(all_df[,style]) < 0.025) / nrow(all_df) * 100, 2)
  lt50 = round(sum(abs(all_df[,style]) < 0.05) / nrow(all_df) * 100, 2)
  lt100 = round(sum(abs(all_df[,style]) < 0.1) / nrow(all_df) * 100, 2)
  
  s = cat("Sparse", ifelse(style=="interp", "Yes", "No"), set, lt10, lt20, lt25, lt50, lt100, sep=" & ")
  cat(s, "\\\\\n")
}

sparse_quantiles_w_se = function(set="train", style="interp") {
  all_df = lapply(all_sparse_filenames, function(x) read.csv(file.path(wdname, set, x)))
  for(i in 1:length(all_df)) {
    all_df[[i]]$round=i
  }
  
  cutoffs = c(0.01, 0.02, 0.025, 0.05, 0.1)
  all_df = Reduce(rbind, all_df)
  s_total = paste("Sparse &", ifelse(style=="interp", "Yes", "No"), "&", set)
  for(cut in cutoffs) {
    res_vec = tapply(all_df[,style], all_df$round, function(x) sum(abs(x) < cut) / length(x))
    mn = round(mean(res_vec) * 100, 2)
    se = round(sd(res_vec) / sqrt(length(res_vec)) * 1.96 * 100, 2)
    s_part = paste(mn, paste("$\\pm ", se, "$", sep=""), sep="")
    s_total = paste(s_total, s_part, sep=" & ")
  }
  
  cat(s_total, "\\\\\n")
}

# Make table without se for main body text
crisp_quantiles(set="train", style="interp")
crisp_quantiles(set="val", style="interp")
crisp_quantiles(set="test", style="interp")
crisp_quantiles(set="train", style="nointerp")
crisp_quantiles(set="val", style="nointerp")
crisp_quantiles(set="test", style="nointerp")

sparse_quantiles(set="train", style="interp")
sparse_quantiles(set="val", style="interp")
sparse_quantiles(set="test", style="interp")
sparse_quantiles(set="train", style="nointerp")
sparse_quantiles(set="val", style="nointerp")
sparse_quantiles(set="test", style="nointerp")

# Make table with se for appendix
crisp_quantiles_w_se(set="train", style="interp")
crisp_quantiles_w_se(set="val", style="interp")
crisp_quantiles_w_se(set="test", style="interp")
crisp_quantiles_w_se(set="train", style="nointerp")
crisp_quantiles_w_se(set="val", style="nointerp")
crisp_quantiles_w_se(set="test", style="nointerp")

sparse_quantiles_w_se(set="train", style="interp")
sparse_quantiles_w_se(set="val", style="interp")
sparse_quantiles_w_se(set="test", style="interp")
sparse_quantiles_w_se(set="train", style="nointerp")
sparse_quantiles_w_se(set="val", style="nointerp")
sparse_quantiles_w_se(set="test", style="nointerp")

###############################
# CRISP MODEL BOUNDARY PLOT   #
###############################

color1 = "#1f77b4"
color2 = "#ff7f0e"
color3 = "#2ca02c"

crisp_all_df_test = lapply(all_crisp_filenames, FUN=function(x) read.csv(file.path(wdname, "test", x)))
for(i in 1:length(crisp_all_df_test)) {
  crisp_all_df_test[[i]]$round = i
  crisp_all_df_test[[i]]$fileno = 1:nrow(crisp_all_df_test[[i]])
}

boundaries = seq(0, 100, length.out=1000)

interp_cdfs = lapply(crisp_all_df_test,
       FUN=function(x) ecdf(abs(x$interp) * 1000))
all_interp_boundaries = lapply(interp_cdfs, function(x) x(boundaries))
interp_cdf = rowMeans(data.frame(all_interp_boundaries))

plot(boundaries, interp_cdf, xlim=c(0, 100), xlab="Boundary error (ms)",
     ylab="Proportion within tolerance", col=color1, type="l", lwd=2, main="Crisp networks")

nointerp_cdfs = lapply(crisp_all_df_test,
                       FUN=function(x) ecdf(abs(x$nointerp) * 1000))
all_nointerp_boundaries = lapply(nointerp_cdfs, function(x) x(boundaries))
nointerp_cdf = rowMeans(data.frame(all_nointerp_boundaries))
lines(boundaries, nointerp_cdf, col=color2, lwd=2)

mfa_test_errs_all = read.csv("../mfa_boundary_eval_res/test/mfa_all_res.csv")
mfa_cdf = ecdf(abs(mfa_test_errs_all$err) * 1000)(boundaries)
lines(boundaries, mfa_cdf, col=color3, lw=2)

legend("bottomright", legend=c("Crisp", "Sparse", "MFA"), col=c(color1, color2, color3), lty=1, lwd=2)

errs_df = data.frame(boundary=boundaries, interp=interp_cdf, nointerp=nointerp_cdf, mfa=mfa_cdf)
write.csv(errs_df, file="test_errs.csv")
system("julia crisp_plot.jl")

###############################
# SPARSE MODEL BOUNDARY PLOT  #
###############################

color1 = "#1f77b4"
color2 = "#ff7f0e"
color3 = "#2ca02c"

sparse_all_df_test = lapply(all_sparse_filenames, FUN=function(x) read.csv(file.path(wdname, "test", x)))
for(i in 1:length(sparse_all_df_test)) {
  sparse_all_df_test[[i]]$round = i
  sparse_all_df_test[[i]]$fileno = 1:nrow(sparse_all_df_test[[i]])
}

boundaries = seq(0, 100, length.out=1000)

sparse_interp_cdfs = lapply(sparse_all_df_test,
                     FUN=function(x) ecdf(abs(x$interp) * 1000))
sparse_all_interp_boundaries = lapply(sparse_interp_cdfs, function(x) x(boundaries))
sparse_interp_cdf = rowMeans(data.frame(sparse_all_interp_boundaries))

plot(boundaries, sparse_interp_cdf, xlim=c(0, 100), xlab="Boundary error (ms)",
     ylab="Proportion within tolerance", col=color1, type="l", lwd=2, main="Sparse networks")

sparse_nointerp_cdfs = lapply(sparse_all_df_test,
                            FUN=function(x) ecdf(abs(x$nointerp) * 1000))
sparse_all_nointerp_boundaries = lapply(sparse_nointerp_cdfs, function(x) x(boundaries))
sparse_nointerp_cdf = rowMeans(data.frame(sparse_all_nointerp_boundaries))
lines(boundaries, sparse_nointerp_cdf, col=color2, lwd=2)

mfa_cdf = ecdf(abs(mfa_test_errs_all$err) * 1000)(boundaries)
lines(boundaries, mfa_cdf, col=color3, lw=2)

legend("bottomright", legend=c("Crisp", "Sparse", "MFA"), col=c(color1, color2, color3), lty=1, lwd=2)

sparse_errs_df = data.frame(boundary=boundaries, interp=sparse_interp_cdf, nointerp=sparse_nointerp_cdf, mfa=mfa_cdf)
write.csv(sparse_errs_df, file="sparse_test_errs.csv")
system("julia sparse_plot.jl")

bisegment_analysis = function(fnames, set="test", style="interp", model="crisp") {

  make_crisp_confusion_plot_data = function(fnames, set="test", style="interp") {
    all_df = lapply(fnames, function(x) read.csv(file.path(wdname, set, x)))
    for(i in 1:length(all_df)) {
      all_df[[i]]$round=i
    }
    
    all_df = Reduce(rbind, all_df)
    return(all_df)
  }
  
  all_df = make_crisp_confusion_plot_data(fnames=fnames, set="test", style="interp")
  all_df$idx = rep(1:sum(all_df$round == 1), times=10)
  all_df$interp = all_df$interp * 1000
  all_df$nointerp = all_df$nointerp * 1000
  all_df = all_df[all_df$next_segment != "#",]
  all_df$targ = all_df[,style]
  
  category_df = read.csv("segment_categories.csv")
  
  new_df = merge(all_df, category_df, by.x="segment", by.y="segment")
  colnames(new_df)[colnames(new_df) == "category"] = "segment_category"
  new_df = merge(new_df, category_df, by.x="next_segment", by.y="segment")
  colnames(new_df)[colnames(new_df) == "category"] = "next_segment_category"
  
  agg = aggregate(targ ~ segment_category + next_segment_category + round,
            data=new_df, FUN=function(x) median(abs(x)))
  colnames(agg)[colnames(agg) == "targ"] = "err"
  
  agg_mn = aggregate(err ~ segment_category + next_segment_category,
                     data=agg, FUN=mean)
  
  outname = paste(set, style, model, "by_category_means.csv", sep="_")
  write.csv(agg_mn, outname, row.names=FALSE, quote=FALSE)
  system(paste("julia make_heatmap.jl", outname, sep=" "))
  
  agg_se = aggregate(err ~ segment_category + next_segment_category,
                     data=agg, FUN=function(x) sd(x) / sqrt(length(x)))
  
  make_category_table = function(x_mn, x_se) {
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
        se = x_se$err[x_se$segment_category == s1 & x_se$next_segment_category == s2]
        se = as.character(round(se * 1.96, digits=2))
        i1 = which(nxt_sgs == s1)
        i2 = which(nxt_sgs == s2) + 1
        d[i1, i2] = paste(mn, "\\pm", se, sep=" ")
        d[i1, i2] = paste("$", d[i1, i2], "$", sep="")
        if(s1 == "silence" && s2 == "silence") d[i1, i2] = "---"
      }
    }
    
    colnames(d)[1] = "Segment"
    d$Segment = cap_sgs
    colnames(d)[2:8] = cap_sgs
    return(d)
  }
  tb = make_category_table(agg_mn, agg_se)
  cat(paste(colnames(tb), collapse=" & ")); cat(" \\\\\n")
  for(i in 1:nrow(tb)) {
    r = tb[i,]
    s = paste(r, collapse=" & ", sep="")
    s = paste(s, " \\\\\n", sep="")
    cat(s)
  }
}

bisegment_analysis(all_crisp_filenames, set="test", style="interp", model="crisp")
bisegment_analysis(all_crisp_filenames, set="test", style="nointerp", model="crisp")

bisegment_analysis(all_sparse_filenames, set="test", style="interp", model="sparse")
bisegment_analysis(all_sparse_filenames, set="test", style="nointerp", model="sparse")
