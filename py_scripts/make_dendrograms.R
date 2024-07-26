# Save as PDFs at 7 in by 3 in (w by h)

setwd("C:/Users/Matt/MAPS_Paper_Code/py_scripts")

d_train = read.csv("train_sparse_distances.csv", header=FALSE)
colnames(d_train) = labs
h_train = hclust(as.dist(d_train))
pdf("train_sparse_dendro.pdf", width=7, height=3)
plot(h_train, yaxt="n", ylab="", main="Train", xlab="", sub=""); dev.off()

d_val = read.csv("val_sparse_distances.csv", header=FALSE)
labs = read.table("phone_labels.txt")[,1]
colnames(d_val) = labs
h_val = hclust(as.dist(d_val))
pdf("val_sparse_dendro.pdf", width=7, height=3)
plot(h_val, yaxt="n", ylab="", main="Val", xlab="", sub=""); dev.off()
