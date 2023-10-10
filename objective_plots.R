library(ggplot2)

setwd("C:/Users/mckelley/alignerv2")

### Crisp network evaluations

frame_names = list.files(path=".", pattern="^real(.)*.txt")
frames = lapply(frame_names, function(x) read.table(x, header=TRUE))
t_losses = lapply(frames, function(x) x$train_loss)
t_losses = Reduce(cbind, t_losses)
t_l_mu = rowMeans(t_losses)
t_l_sig = apply(t_losses, 1, sd)
t_l_se = t_l_sig / sqrt(10)
t_l_df = data.frame(epoch=1:50,
                    loss=t_l_mu,
                    selow=(t_l_mu - 1.96*t_l_se),
                    sehigh=(t_l_mu + 1.96*t_l_se))
ggplot(t_l_df, aes(epoch, main="Train loss")) +
  geom_line(aes(y=loss), col="blue") +
  ylab("Average train loss") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

v_losses = lapply(frames, function(x) x$val_loss)
v_losses = Reduce(cbind, v_losses)
v_l_mu = rowMeans(v_losses)
v_l_sig = apply(v_losses, 1, sd)
v_l_se = v_l_sig / sqrt(10)
v_l_df = data.frame(epoch=1:50,
                    loss=v_l_mu,
                    selow=(v_l_mu - 1.96*v_l_se),
                    sehigh=(v_l_mu + 1.96*v_l_se))
ggplot(v_l_df, aes(epoch)) +
  geom_line(aes(y=loss), col="blue") + 
  ylab("Average val loss") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

losses = rbind(t_l_df, v_l_df)
losses$type = rep(c("Train", "Val"), each=50)

ggplot(losses, aes(epoch, loss, ymin=selow, ymax=sehigh, group=type, color=type)) +
  geom_line(size=0.75) +
  geom_ribbon(alpha=0.2, color=NA) +
  xlab("Epoch") +
  ylab("Cross-entropy loss") +
  labs(color="Dataset")

t_accs = lapply(frames, function(x) x$train_acc)
t_accs = Reduce(cbind, t_accs)
t_a_mu = rowMeans(t_accs)
t_a_sig = apply(t_accs, 1, sd)
t_a_se = t_a_sig / sqrt(10)
t_a_df = data.frame(epoch=1:50,
                    acc=t_a_mu,
                    selow=(t_a_mu - 1.96*t_a_se),
                    sehigh=(t_a_mu + 1.96*t_a_se))
ggplot(t_a_df, aes(epoch, main="Train loss")) +
  geom_line(aes(y=acc), col="blue") +
  ylab("Average train balanced accuracy") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

v_accs = lapply(frames, function(x) x$val_acc)
v_accs = Reduce(cbind, v_accs)
v_a_mu = rowMeans(v_accs)
v_a_sig = apply(v_accs, 1, sd)
v_a_se = v_a_sig / sqrt(10)
v_a_df = data.frame(epoch=1:50,
                    acc=v_a_mu,
                    selow=(v_a_mu - 1.96*v_a_se),
                    sehigh=(v_a_mu + 1.96*v_a_se))
ggplot(v_a_df, aes(epoch, main="Train acc")) +
  geom_line(aes(y=acc), col="blue") +
  ylab("Average train accuracy") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

accs = rbind(t_a_df, v_a_df)
accs$type = rep(c("Train", "Val"), each=50)

ggplot(accs, aes(epoch, acc, ymin=selow, ymax=sehigh, group=type, color=type)) +
  geom_line(size=0.75) +
  geom_ribbon(alpha=0.2, color=NA) +
  xlab("Epoch") +
  ylab("Accuracy") +
  labs(color="Dataset")


### Tagger network evaluations

frame_names = list.files(path=".", pattern="full(.)*.txt")
frames = lapply(frame_names, function(x) read.table(x, header=TRUE))
t_losses = lapply(frames, function(x) x$train_loss)
t_losses = Reduce(cbind, t_losses)
t_l_mu = rowMeans(t_losses)
t_l_sig = apply(t_losses, 1, sd)
t_l_se = t_l_sig / sqrt(10)
t_l_df = data.frame(epoch=1:50,
                    loss=t_l_mu,
                    selow=(t_l_mu - 1.96*t_l_se),
                    sehigh=(t_l_mu + 1.96*t_l_se))
ggplot(t_l_df, aes(epoch, main="Train loss")) +
  geom_line(aes(y=loss), col="blue") +
  ylab("Average train loss") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

v_losses = lapply(frames, function(x) x$val_loss)
v_losses = Reduce(cbind, v_losses)
v_l_mu = rowMeans(v_losses)
v_l_sig = apply(v_losses, 1, sd)
v_l_se = v_l_sig / sqrt(10)
v_l_df = data.frame(epoch=1:50,
                    loss=v_l_mu,
                    selow=(v_l_mu - 1.96*v_l_se),
                    sehigh=(v_l_mu + 1.96*v_l_se))
ggplot(v_l_df, aes(epoch)) +
  geom_line(aes(y=loss), col="blue") + 
  ylab("Average val loss") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

losses = rbind(t_l_df, v_l_df)
losses$type = rep(c("Train", "Val"), each=50)

ggplot(losses, aes(epoch, loss, ymin=selow, ymax=sehigh, group=type, color=type)) +
  geom_line(size=0.75) +
  geom_ribbon(alpha=0.2, color=NA) +
  xlab("Epoch") +
  ylab("Weighted cross-entropy loss") +
  labs(color="Dataset")

t_accs = lapply(frames, function(x) x$train_balacc)
t_accs = Reduce(cbind, t_accs)
t_a_mu = rowMeans(t_accs)
t_a_sig = apply(t_accs, 1, sd)
t_a_se = t_a_sig / sqrt(10)
t_a_df = data.frame(epoch=1:50,
                    acc=t_a_mu,
                    selow=(t_a_mu - 1.96*t_a_se),
                    sehigh=(t_a_mu + 1.96*t_a_se))
ggplot(t_a_df, aes(epoch, main="Train loss")) +
  geom_line(aes(y=acc), col="blue") +
  ylab("Average train balanced accuracy") +
geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

v_accs = lapply(frames, function(x) x$val_balacc)
v_accs = Reduce(cbind, v_accs)
v_a_mu = rowMeans(v_accs)
v_a_sig = apply(v_accs, 1, sd)
v_a_se = v_a_sig / sqrt(10)
v_a_df = data.frame(epoch=1:50,
                    acc=v_a_mu,
                    selow=(v_a_mu - 1.96*v_a_se),
                    sehigh=(v_a_mu + 1.96*v_a_se))
ggplot(v_a_df, aes(epoch, main="Train loss")) +
  geom_line(aes(y=acc), col="blue") +
  ylab("Average train balanced accuracy") +
  geom_ribbon(aes(ymin=selow, ymax=sehigh), alpha=0.2)

accs = rbind(t_a_df, v_a_df)
accs$type = rep(c("Train", "Val"), each=50)

ggplot(accs, aes(epoch, acc, ymin=selow, ymax=sehigh, group=type, color=type)) +
  geom_line(size=0.75) +
  geom_ribbon(alpha=0.2, color=NA) +
  xlab("Epoch") +
  ylab("Balanced accuracy") +
  labs(color="Dataset")
