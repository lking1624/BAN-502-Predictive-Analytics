library(tidyverse)
library(tidymodels)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)

heart <- read_csv("heart_disease-1.csv")

heart_cleaned <- heart |>
  mutate(Sex = as_factor(Sex)) |>
  mutate(ChestPainType = as_factor(ChestPainType)) |>
  mutate(RestingECG = as_factor(RestingECG)) |>
  mutate(ExerciseAngina = factor(ExerciseAngina)) |>
  mutate(ST_Slope = factor(ST_Slope)) |>
  mutate(HeartDisease = as_factor(HeartDisease)) |>
  mutate(HeartDisease = fct_recode(HeartDisease, "No" = "0", "Yes" = "1"))


# Q1
set.seed(12345)
training_split <- initial_split(
  heart_cleaned,
  prop = 0.7,
  strata = HeartDisease
)
heart_train <- training(training_split) # 642 Rows
heart_test <- testing(training_split) # 276 Rows
# Q2 The first split in the tree is a split on which variable? ST_Slope
heart_recipe <- recipe(HeartDisease ~ ., heart_train)

tree_model <- decision_tree() |>
  set_engine("rpart", model = TRUE) |> #don't forget the model = TRUE flag
  set_mode("classification")

heart_workflow <-
  workflow() |>
  add_model(tree_model) |>
  add_recipe(heart_recipe)

heart_fit <- fit(heart_workflow, heart_train)
heart_fit |>
  pull_workflow_fit() |>
  pluck("fit")
tree <- heart_fit %>%
  pull_workflow_fit() %>%
  pluck("fit")

#plot the tree
rpart.plot(tree)

# Q3 examine the complexity parameter (cp) values tried by R.
# Which cp value is optimal (recall that the optimal cp corresponds to the minimized “xerror” value)? Report
# your answer to two decimal places.
heart_fit$fit$fit$fit$cptable
0.02

# Q4
set.seed(123)
folds = vfold_cv(heart_train, v = 5)

heart_recipe <- recipe(HeartDisease ~ ., heart_train) %>%
  step_dummy(all_nominal(), -all_outcomes())

tree_model <- decision_tree(cost_complexity = tune()) %>%
  set_engine("rpart", model = TRUE) %>% #don't forget the model = TRUE flag
  set_mode("classification")

tree_grid = grid_regular(cost_complexity(), levels = 25) #try 25 sensible values for cp

heart_workflow =
  workflow() %>%
  add_model(tree_model) %>%
  add_recipe(heart_recipe)

tree_res =
  heart_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tree_grid
  )

tree_res

tree_res %>%
  collect_metrics() %>%
  ggplot(aes(cost_complexity, mean)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~.metric, scales = "free", nrow = 2)

best_tree <- tree_res |>
  select_best(metric = "accuracy")

best_tree

final_wf <-
  heart_workflow |>
  finalize_workflow(best_tree)

final_fit <- fit(final_wf, heart_train)

tree <- final_fit |>
  pull_workflow_fit() |>
  pluck("fit")

fancyRpartPlot(tree, tweak = 1.2)

treepred = predict(final_fit, heart_train, type = "class")
head(treepred)

confusionMatrix(
  treepred$.pred_class,
  heart_train$HeartDisease,
  positive = "Yes"
)


treepred_test = predict(final_fit, heart_test, type = "class")
head(treepred_test)

confusionMatrix(
  treepred_test$.pred_class,
  heart_test$HeartDisease,
  positive = "Yes"
)
