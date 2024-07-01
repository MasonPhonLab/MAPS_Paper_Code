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
  
  md_str = paste("$", round(mean(mdae) * 1000, 2), " \\pm ", round(1.96 * md_sterr, 2), " \\, \\text{ms}$", sep="")
  mn_str = paste("$", round(mean(mnae) * 1000, 2), " \\pm ", round(1.96 * mn_sterr, 2), " \\, \\text{ms}$", sep="")
  
  cat(paste("Crisp", ifelse(style == "interp", "Yes", "No"), set, mn_str, md_str, sep=" & "), "\\\\")
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
  
  md_str = paste("$", round(mean(mdae) * 1000, 2), " \\pm ", round(1.96 * md_sterr, 2), " \\, \\text{ms}$", sep="")
  mn_str = paste("$", round(mean(mnae) * 1000, 2), " \\pm ", round(1.96 * mn_sterr, 2), " \\, \\text{ms}$", sep="")
  
  cat(paste("Sparse", ifelse(style == "interp", "Yes", "No"), set, mn_str, md_str, sep=" & "), "\\\\")
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
  cat(s, "\\\\")
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
  
  cat(s_total, "\\\\")
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
  cat(s, "\\\\")
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
  
  cat(s_total, "\\\\")
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
mfa_cdf = ecdf(abs(mfa_test_errs_all$errs) * 1000)(boundaries)
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

mfa_cdf = ecdf(abs(mfa_test_errs_all$errs) * 1000)(boundaries)
lines(boundaries, mfa_cdf, col=color3, lw=2)

legend("bottomright", legend=c("Crisp", "Sparse", "MFA"), col=c(color1, color2, color3), lty=1, lwd=2)

sparse_errs_df = data.frame(boundary=boundaries, interp=sparse_interp_cdf, nointerp=sparse_nointerp_cdf, mfa=mfa_cdf)
write.csv(sparse_errs_df, file="sparse_test_errs.csv")
system("julia sparse_plot.jl")
