SCRAP


#Setting Empty spaces to NA
train[train == ""] <- NA

NA_index <- apply(train, 2,function(x){which(sum(is.na(x))>0)})

##67 is equal so no all rows are 2%
NA_col <- colnames(train)[colSums(is.na(train)) > 0]
NAdf <- train[,NA_col]
out <- NAdf[apply(!is.na(NAdf), 1, any), ]

#Removing 406 Observation which are spurious and due to error
ctrain <- train[-as.numeric(row.names(out)),]
dim(ctrain)

#Removing NA columns
btrain <- ctrain[colSums(!is.na(ctrain)) > 0]

