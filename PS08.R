library(tidyverse)
library(caret)
library(tictoc)

# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/PS08/train.csv")


# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:
n_values <- seq(from=100000, to=5000000, by=100000)
k_values <- c(1, seq(from=25, to=200, by=25))

# Time knn here -----------------------------------------------------------
# Set seed
set.seed(495)
# Initialize empty dataframe
df <- data.frame(
  n = numeric(),
  k = numeric(),
  time = numeric()
)
count = 1   # data frame row counter
# Time training time for all values in n_values, k_values
for (n in n_values){
  train_samp <- sample_n(train, n)
  for (k in k_values){
    tic()
    model_knn <- caret::knn3(model_formula, data=train_samp, k = k)
    timer_info <- toc()
    time <- timer_info$toc - timer_info$tic
    
    tuple <- c(n, k, time)
    df[count,] <- tuple
    count <- count + 1
  }
}
# Plot results
(runtime_plot <- ggplot(df, aes(x=n, y=time, col=k, group=k)) +
  geom_line() +
  scale_colour_gradientn(colors=c("blue", "orange")) +
  labs(
    title="Training time of kNN with different k and training set sizes",
    x="Number of training observations",
    y="Time to train model (seconds)",
    color="Value of k"
  ))
ggsave(filename="jonathan_che.png", width=16, height = 9)

# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# -k: number of neighbors to consider
# -d: number of predictors used? In this case d is fixed at 3


