## only have 21840 aligned files in train set
## only have 3886 aligned files in val set
## only have 6417 aligned files in test set

wdname = "C:/Users/Matt/APhL_Aligner_Paper_Code/py_scripts/mfa_boundary_eval_res"
setwd(wdname)

mfa_row = function(set="train") {
  errs = read.csv(file.path(wdname, set, "mfa_all_res.csv"))[,1]
  mdae = round(median(abs(errs)) * 1000, 2)
  mnae = round(mean(abs(errs)) * 1000, 2)
  
  md_str = paste(mdae, "ms")
  mn_str = paste(mnae, "ms")
  
  cat(paste("MFA", "---", set, mn_str, md_str, sep=" & "), "\\\\")
}

mfa_row("train")
mfa_row("val")
mfa_row("test")

mfa_quantiles = function(set="train") {
  errs = abs(read.csv(file.path(wdname, set, "mfa_all_res.csv"))[,1])
  
  lt10 = round(sum(errs < 0.01) / length(errs) * 100, 2)
  lt20 = round(sum(errs < 0.02) / length(errs) * 100, 2)
  lt25 = round(sum(errs < 0.025) / length(errs) * 100, 2)
  lt50 = round(sum(errs < 0.05) / length(errs) * 100, 2)
  lt100 = round(sum(errs < 0.1) / length(errs) * 100, 2)
  
  cat("MFA", "---", set, lt10, lt20, lt25, lt50, lt100, sep=" & ")
}

mfa_quantiles(set="train")
mfa_quantiles(set="val")
mfa_quantiles(set="test")
