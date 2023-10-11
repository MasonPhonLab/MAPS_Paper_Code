##### CRISP
### AVG. NUM. EPOCHS
dat_path = "performance_metrics"
crispdats = lapply(list.files(path=dat_path, pattern="^real"),
                   function(x) read.table(file.path(dat_path, x), header=TRUE))
best_crisp_epochs = sapply(crispdats,
                           function(x) {
                             mx = max(x$val_acc)
                             which(x$val_acc == mx)
                           })
print(best_crisp_epochs)
mean(best_crisp_epochs)
sd(best_crisp_epochs) / sqrt(length(best_crisp_epochs) - 1)

best_crisp_train_loss = sapply(1:10, function(i)
  crispdats[[i]]$train_loss[best_crisp_epochs[i]]) 
mean(best_crisp_train_loss)
1.96 * sd(best_crisp_train_loss) / sqrt(10-1)

best_crisp_train_acc = sapply(1:10, function(i)
  crispdats[[i]]$train_acc[best_crisp_epochs[i]])
mean(best_crisp_train_acc)
1.96 * sd(best_crisp_train_acc) / sqrt(10-1)

best_crisp_val_loss = sapply(1:10, function(i)
  crispdats[[i]]$val_loss[best_crisp_epochs[i]]) 
mean(best_crisp_val_loss)
1.96 * sd(best_crisp_val_loss) / sqrt(10-1)

best_crisp_val_acc = sapply(1:10, function(i)
  crispdats[[i]]$val_acc[best_crisp_epochs[i]])
mean(best_crisp_val_acc)
1.96 * sd(best_crisp_val_acc) / sqrt(10-1)

crisp_test_dat = read.table("performance_metrics/crisp_test_res.txt", header=TRUE)
mean(crisp_test_dat$loss)
1.96 * sd(crisp_test_dat$loss) / sqrt(10-1)

mean(crisp_test_dat$accuracy)
1.96 * sd(crisp_test_dat$accuracy) / sqrt(10-1)

###### SPARSE
### AVG. NUM. EPOCHS

sparsedats = lapply(list.files(path="performance_metrics", pattern="full"),
                    function(x) read.table(file.path("performance_metrics", x), header=TRUE))
best_sparse_epochs = sapply(sparsedats,
                            function (x) {
                              mx = max(x$val_balacc)
                              which(x$val_balacc == mx)
                            })
print(best_sparse_epochs)
mean(best_sparse_epochs)
sd(best_sparse_epochs) / sqrt(10-1)

frames = lapply(list.files(path="performance_metrics", pattern="^full"),
                function(x) read.table(file.path("performance_metrics", x), header=TRUE))

### Train metrics
all_train_loss = lapply(frames, function(x) x$train_loss)
best_train_loss = sapply(1:10, function(x)
  all_train_loss[[x]][best_sparse_epochs[x]])
mean(best_train_loss)
1.96 * sd(best_train_loss) / sqrt(10-1)

all_train_sens = lapply(frames, function(x) x$train_sens)
best_train_sens = sapply(1:10, function(x)
  all_train_sens[[x]][best_sparse_epochs[x]])
mean(best_train_sens)
1.96 * sd(best_train_sens) / sqrt(length(best_train_sens) - 1)

all_train_spec = lapply(frames, function(x) x$train_spec)
best_train_spec = sapply(1:10, function(x)
  all_train_spec[[x]][best_sparse_epochs[x]])
mean(best_train_spec)
1.96 * sd(best_train_spec) / sqrt(length(best_train_spec) - 1)

all_train_balacc = lapply(frames, function(x) x$train_balacc)
best_train_balacc = sapply(1:10, function(x)
  all_train_balacc[[x]][best_sparse_epochs[x]])
mean(best_train_balacc)
1.96 * sd(best_train_balacc) / sqrt(length(best_train_balacc) - 1)


### Val. metrics

all_val_loss = lapply(frames, function(x) x$val_loss)
best_val_loss = sapply(1:10, function(x)
  all_val_loss[[x]][best_sparse_epochs[x]])
mean(best_val_loss)
1.96 * sd(best_val_loss) / sqrt(10-1)

all_val_sens = lapply(frames, function(x) x$val_sens)
best_val_sens = sapply(1:10, function(x)
  all_val_sens[[x]][best_sparse_epochs[x]])
mean(best_val_sens)
1.96 * sd(best_val_sens) / sqrt(length(best_val_sens) - 1)

all_val_spec = lapply(frames, function(x) x$val_spec)
best_val_spec = sapply(1:10, function(x)
  all_val_spec[[x]][best_sparse_epochs[x]])
mean(best_val_spec)
1.96 * sd(best_val_spec) / sqrt(length(best_val_spec) - 1)

all_val_balacc = lapply(frames, function(x) x$val_balacc)
best_val_balacc = sapply(1:10, function(x)
  all_val_balacc[[x]][best_sparse_epochs[x]])
mean(best_val_balacc)
1.96 * sd(best_val_balacc) / sqrt(length(best_val_balacc) - 1)

### Test metrics

sparse_test = read.table("performance_metrics/sparse_test_res.txt", header=TRUE, sep="\t")

mean(sparse_test$loss)
1.96 * sd(sparse_test$loss) / sqrt(10-1)

mean(sparse_test$sens)
1.96 * sd(sparse_test$sens) / sqrt(nrow(sparse_test) - 1)

mean(sparse_test$spec)
1.96 * sd(sparse_test$spec) / sqrt(nrow(sparse_test) - 1)

mean(sparse_test$balacc)
1.96 * sd(sparse_test$balacc) / sqrt(nrow(sparse_test) - 1)

