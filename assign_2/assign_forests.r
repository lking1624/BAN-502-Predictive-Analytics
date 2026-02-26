## ── Libraries ─────────────────────────────────────────────────────────────────
library(tidyverse)
library(tidymodels)
library(caret)
library(gridExtra)
library(vip)
library(ranger)

## ── Load & Clean Data ────────────────────────────────────────────────────────
drug <- read_csv(
  "drug_data-2.csv",
  col_names = FALSE,
  skip = 1,
  show_col_types = FALSE
)

names(drug) <- c(
  "ID",
  "Age",
  "Gender",
  "Education",
  "Country",
  "Ethnicity",
  "Nscore",
  "Escore",
  "Oscore",
  "Ascore",
  "Cscore",
  "Impulsive",
  "SS",
  "Alcohol",
  "Amphet",
  "Amyl",
  "Benzos",
  "Caff",
  "Cannabis",
  "Choc",
  "Coke",
  "Crack",
  "Ecstasy",
  "Heroin",
  "Ketamine",
  "Legalh",
  "LSD",
  "Meth",
  "Mushrooms",
  "Nicotine",
  "Semer",
  "VSA"
)

drug[drug == "CL0"] <- "No"
drug[drug == "CL1"] <- "No"
drug[drug == "CL2"] <- "Yes"
drug[drug == "CL3"] <- "Yes"
drug[drug == "CL4"] <- "Yes"
drug[drug == "CL5"] <- "Yes"
drug[drug == "CL6"] <- "Yes"

drug_clean <- drug %>%
  mutate_at(vars(Age:Ethnicity), funs(as_factor)) %>%
  mutate(
    Age = factor(
      Age,
      labels = c("18_24", "25_34", "35_44", "45_54", "55_64", "65_")
    )
  ) %>%
  mutate(Gender = factor(Gender, labels = c("Male", "Female"))) %>%
  mutate(
    Education = factor(
      Education,
      labels = c(
        "Under16",
        "At16",
        "At17",
        "At18",
        "SomeCollege",
        "ProfessionalCert",
        "Bachelors",
        "Masters",
        "Doctorate"
      )
    )
  ) %>%
  mutate(
    Country = factor(
      Country,
      labels = c(
        "USA",
        "NewZealand",
        "Other",
        "Australia",
        "Ireland",
        "Canada",
        "UK"
      )
    )
  ) %>%
  mutate(
    Ethnicity = factor(
      Ethnicity,
      labels = c(
        "Black",
        "Asian",
        "White",
        "White/Black",
        "Other",
        "White/Asian",
        "Black/Asian"
      )
    )
  ) %>%
  mutate_at(vars(Alcohol:VSA), funs(as_factor)) %>%
  select(-ID)

str(drug_clean)

# Drop all drug columns except Nicotine
drug_clean <- drug_clean %>%
  select(!(Alcohol:Mushrooms)) %>%
  select(!(Semer:VSA))

str(drug_clean)

## ── Question 1: Check for missing data ──────────────────────────────────────
# True/False: There is missingness in the dataset.
colSums(is.na(drug_clean))
# Answer: FALSE — no missing data

## ── Question 2: Train/Test Split (70/30, stratify on Nicotine) ──────────────
set.seed(1234)
drug_split <- initial_split(drug_clean, prop = 0.70, strata = Nicotine)
drug_train <- training(drug_split)
drug_test <- testing(drug_split)

nrow(drug_train)
# Answer: 1318 rows in the training set

## ── Question 3: EDA Visualizations ──────────────────────────────────────────
# Group 1: Factor variables (4 plots)
p1 <- ggplot(drug_train, aes(x = Age, fill = Nicotine)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion") +
  theme_minimal()
p2 <- ggplot(drug_train, aes(x = Gender, fill = Nicotine)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion") +
  theme_minimal()
p3 <- ggplot(drug_train, aes(x = Education, fill = Nicotine)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
p4 <- ggplot(drug_train, aes(x = Country, fill = Nicotine)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

grid.arrange(p1, p2, p3, p4, ncol = 2)

# Group 2: Ethnicity + 3 numeric variables
p5 <- ggplot(drug_train, aes(x = Ethnicity, fill = Nicotine)) +
  geom_bar(position = "fill") +
  labs(y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
p6 <- ggplot(drug_train, aes(x = Nicotine, y = Nscore)) +
  geom_boxplot() +
  theme_minimal()
p7 <- ggplot(drug_train, aes(x = Nicotine, y = Escore)) +
  geom_boxplot() +
  theme_minimal()
p8 <- ggplot(drug_train, aes(x = Nicotine, y = Oscore)) +
  geom_boxplot() +
  theme_minimal()

grid.arrange(p5, p6, p7, p8, ncol = 2)

# Group 3: Remaining numeric variables
p9 <- ggplot(drug_train, aes(x = Nicotine, y = Ascore)) +
  geom_boxplot() +
  theme_minimal()
p10 <- ggplot(drug_train, aes(x = Nicotine, y = Cscore)) +
  geom_boxplot() +
  theme_minimal()
p11 <- ggplot(drug_train, aes(x = Nicotine, y = Impulsive)) +
  geom_boxplot() +
  theme_minimal()
p12 <- ggplot(drug_train, aes(x = Nicotine, y = SS)) +
  geom_boxplot() +
  theme_minimal()

grid.arrange(p9, p10, p11, p12, ncol = 2)

# Q3 Answer: TRUE — 18-24 age group proportionally more likely to be Nicotine users

## ── Question 4 ──────────────────────────────────────────────────────────────
# Q4 Answer: TRUE — higher Impulsive scores are more likely among Nicotine users

## ── Question 5: Random Forest Tuning ────────────────────────────────────────
drug_recipe <- recipe(Nicotine ~ ., data = drug_train)

rf_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 100) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

drug_wflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(drug_recipe)

set.seed(123)
drug_folds <- vfold_cv(drug_train, v = 5)

rf_grid <- grid_regular(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  levels = 10
)

set.seed(123)
rf_res <- tune_grid(
  drug_wflow,
  resamples = drug_folds,
  grid = rf_grid
)

autoplot(rf_res)
# Q5 Answer: B. 0.730 — highest accuracy is just greater than 0.725

## ── Question 6: Finalize Model & Variable Importance ────────────────────────
best_rf <- select_best(rf_res, metric = "accuracy")
best_rf

final_wflow <- finalize_workflow(drug_wflow, best_rf)
final_fit <- fit(final_wflow, data = drug_train)

final_fit %>%
  extract_fit_parsnip() %>%
  vip(geom = "point")
# Q6 Answer: D. SS

## ── Question 7: Training Set Accuracy ───────────────────────────────────────
train_preds <- predict(final_fit, drug_train)
train_results <- bind_cols(drug_train, train_preds)

confusionMatrix(
  train_results$.pred_class,
  train_results$Nicotine,
  positive = "Yes"
)
# Q7 Answer: 0.9484

## ── Question 8: Naive Accuracy ──────────────────────────────────────────────
drug_train %>%
  count(Nicotine) %>%
  mutate(prop = n / sum(n))
# Q8 Answer: 0.6707

## ── Question 9: Testing Set Accuracy ────────────────────────────────────────
test_preds <- predict(final_fit, drug_test)
test_results <- bind_cols(drug_test, test_preds)

confusionMatrix(
  test_results$.pred_class,
  test_results$Nicotine,
  positive = "Yes"
)
# Q9 Answer: 0.7108

## ── Question 10: Overfitting Assessment ─────────────────────────────────────
# Training accuracy: 0.9484
# Testing accuracy:  0.7108
# Large gap (0.24) suggests the model memorized training data.
# Q10 Answer: B. Overfitting is likely occurring
