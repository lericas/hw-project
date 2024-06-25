# List of packages to be loaded
packages <- c("tidyverse", "ggplot2", "ggthemes", "data.table", "lubridate",
              "caret", "knitr", "scales", "treemapify", "dplyr", "recommenderlab",
              "xgboost")

# Check and install packages if not already installed
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
}

# Load necessary libraries
invisible(sapply(packages, library, character.only = TRUE))

# Set the timeout option to 120 seconds to allow for large file downloads
options(timeout = 120)

# Create a temporary file for the download
dl <- tempfile()

# Download the MovieLens dataset
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Define the paths for the ratings and movies files
ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"

# Unzip the ratings file if it doesn't already exist
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

# Unzip the movies file if it doesn't already exist
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Read the ratings data and split into columns
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

# Convert columns to appropriate data types
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read the movies data and split into columns
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")

# Convert columns to appropriate data types
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merge the ratings and movies dataframes by movieId
movielens <- left_join(ratings, movies, by = "movieId")

# Set a random seed for reproducibility and create a test set (10% of data)
set.seed(1, sample.kind="Rounding") # Use this if using R 3.6 or later
# set.seed(1) # Use this if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure userId and movieId in the final hold-out test set are also in the edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from the final hold-out test set back into the edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up the workspace by removing unnecessary objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split edx data into training and validation sets
set.seed(1, sample.kind = "Rounding")
train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.9, list = FALSE)
train_set <- edx[train_index, ]
validation_set <- edx[-train_index, ]

# Check the structure of the training set
glimpse(train_set)

# Summary statistics of the ratings
summary(train_set$rating)

# Distribution of ratings
ggplot(train_set, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
  ggtitle("Distribution of Ratings") +
  xlab("Rating") +
  ylab("Count")

# Number of unique users and movies
num_users <- n_distinct(train_set$userId)
num_movies <- n_distinct(train_set$movieId)
cat("Number of unique users:", num_users, "\n")
cat("Number of unique movies:", num_movies, "\n")

# Average rating per movie
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(avg_rating = mean(rating))


# Average rating per user
user_avgs <- train_set %>%
  group_by(userId) %>%
  summarize(avg_rating = mean(rating))

# Plot average ratings per movie
ggplot(movie_avgs, aes(x = avg_rating)) +
  geom_histogram(binwidth = 0.1, fill = "green", color = "black") +
  ggtitle("Average Rating per Movie") +
  xlab("Average Rating") +
  ylab("Count")

# Plot average ratings per user
ggplot(user_avgs, aes(x = avg_rating)) +
  geom_histogram(binwidth = 0.1, fill = "red", color = "black") +
  ggtitle("Average Rating per User") +
  xlab("Average Rating") +
  ylab("Count")

# Calculate mean ratings for each movie
mean_ratings <- train_set %>%
  group_by(movieId) %>%
  summarise(mean_rating = mean(rating, na.rm = TRUE))

# Merge mean ratings with validation_set to predict ratings
predictions_mean <- merge(validation_set, mean_ratings, by = "movieId", all.x = TRUE)

# Calculate RMSE for mean-based model
rmse_mean <- sqrt(mean((predictions_mean$rating - predictions_mean$mean_rating)^2, na.rm = TRUE))
print(paste("RMSE for Mean-Based Model on validation_set:", rmse_mean))

# Model Mean Rating
mean_rating <- mean(train_set$rating)

# Predict ratings using the mean rating
predictions_mean <- rep(mean_rating, nrow(validation_set))

# Calculate RMSE for the mean rating model on validation set
rmse_mean <- RMSE(predictions_mean, validation_set$rating)
cat("RMSE for mean rating model (validation set):", rmse_mean, "\n")

# Predict ratings using the mean rating for final hold-out test set
predictions_mean_holdout <- rep(mean_rating, nrow(final_holdout_test))

# Calculate RMSE for the mean rating model on final hold-out test set
rmse_mean_holdout <- RMSE(predictions_mean_holdout, final_holdout_test$rating)
cat("RMSE for mean rating model (final hold-out test set):", rmse_mean_holdout, "\n")

# Model - Linear Regression
lm_model <- lm(rating ~ movieId + userId, data = train_set)

# Predict ratings using the linear regression model on validation set
predictions_lm <- predict(lm_model, validation_set)

# Calculate RMSE for the linear regression model on validation set
rmse_lm <- RMSE(predictions_lm, validation_set$rating)
cat("RMSE for linear regression model (validation set):", rmse_lm, "\n")

# Predict ratings using the linear regression model on final hold-out test set
predictions_lm_holdout <- predict(lm_model, final_holdout_test)

# Calculate RMSE for the linear regression model on final hold-out test set
rmse_lm_holdout <- RMSE(predictions_lm_holdout, final_holdout_test$rating)
cat("RMSE for linear regression model (final hold-out test set):", rmse_lm_holdout, "\n")

# Model - Prepare data for xgboost
train_matrix <- xgb.DMatrix(data = as.matrix(train_set %>% select(userId, movieId)),
                            label = train_set$rating)
validation_matrix <- xgb.DMatrix(data = as.matrix(validation_set %>% select(userId, movieId)),
                                 label = validation_set$rating)
final_holdout_matrix <- xgb.DMatrix(data = as.matrix(final_holdout_test %>% select(userId, movieId)),
                                    label = final_holdout_test$rating)

# Set parameters for xgboost
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 5,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model
set.seed(1)
xgb_model <- xgboost(data = train_matrix, params = params, nrounds = 100, verbose = 0)

# Predict and calculate RMSE for validation set
predictions_xgb <- predict(xgb_model, validation_matrix)
rmse_xgb <- RMSE(predictions_xgb, validation_set$rating)
cat("RMSE for XGBoost model (validation set):", rmse_xgb, "\n")


# Predict and calculate RMSE for final hold-out test set
predictions_xgb_holdout <- predict(xgb_model, final_holdout_matrix)
rmse_xgb_holdout <- RMSE(predictions_xgb_holdout, final_holdout_test$rating)
cat("RMSE for XGBoost model (final hold-out test set):", rmse_xgb_holdout, "\n")

# Model -  Calculate global mean rating
global_mean <- mean(train_set$rating)

# Calculate user biases
user_biases <- train_set %>%
  group_by(userId) %>%
  summarize(user_bias = mean(rating - global_mean))

# Calculate item biases
item_biases <- train_set %>%
  group_by(movieId) %>%
  summarize(item_bias = mean(rating - global_mean - user_biases$user_bias))

# Predict ratings on validation set using bias subtraction
predictions_bias <- validation_set %>%
  left_join(user_biases, by = "userId") %>%
  left_join(item_biases, by = "movieId") %>%
  mutate(predicted_rating = global_mean + user_bias + item_bias) %>%
  pull(predicted_rating)

# Calculate RMSE for bias subtraction model on validation set
rmse_bias <- RMSE(predictions_bias, validation_set$rating)
cat("RMSE for bias subtraction model (validation set):", rmse_bias, "\n")

# Predict ratings for final hold-out test set using bias subtraction
predictions_bias_holdout <- final_holdout_test %>%
  left_join(user_biases, by = "userId") %>%
  left_join(item_biases, by = "movieId") %>%
  mutate(predicted_rating = global_mean + user_bias + item_bias) %>%
  pull(predicted_rating)

# Calculate RMSE for bias subtraction model on final hold-out test set
rmse_bias_holdout <- RMSE(predictions_bias_holdout, final_holdout_test$rating)
cat("RMSE for bias subtraction model (final hold-out test set):", rmse_bias_holdout, "\n")
