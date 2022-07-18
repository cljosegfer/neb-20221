rm(list=ls())

library(Rcpp)

sourceCpp('GGClassification_gabriel_graph.cpp')

# params
dir_matlab_file <- 'data/comvoi-en'
dir_gg <- 'data/gg'
K <- 10

# processo
for (fold_n in seq(K)){
  # read
  filename <- sprintf('%s/comvoi-en_fold_%s.mat',
                      dir_matlab_file, fold_n - 1)
  data_mat <- R.matlab::readMat(filename)
  
  # train / test
  train <- data_mat$X.train
  # class_train <- data_mat$y.train
  # test <- data_mat$X.test
  # class_test <- data_mat$y.test

  # gg
  gg <- GabrielGraph(train)
  
  # write
  path <- sprintf('%s/comvoi-en_fold_%s.csv',
                  dir_gg, fold_n - 1)
  write.csv(gg, path, row.names = FALSE)

  # log
  log <- sprintf('fold : %s / %s', fold_n, K)
  print(log)
}
