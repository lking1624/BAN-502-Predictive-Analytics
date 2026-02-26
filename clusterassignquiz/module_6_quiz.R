library(tidyverse)
library(cluster)
library(factoextra)
library(dendextend)

truck <- read_csv("trucks-1.csv") |>
  mutate(Driver_ID = as.character(Driver_ID))

# Question 1: Plot the relationship between Distance and Speed
ggplot(truck, aes(Distance, Speeding)) +
  geom_point()
# The longer the distance, the more likely the driver is to speed.

# Create a new data frame called “trucks_cleaned” that contains the scaled and centered variables.
# Two notes: 1) The “predictor” variables in the recipe are “Distance” and “Speeding” and 2) There is no need
# to create dummy variables as there are no categorical variables in the data. Be sure that you do NOT include
# the Driver_ID variable.
# What is the maximum value (to four decimal places) of the Distance variable in the scaled dataset?
truck_cleaned <- truck |>
  select(-Driver_ID) |>
  scale() |>
  as.data.frame()
round(max(truck_cleaned$Distance), 4)


# Use k-Means clustering with two clusters (k=2) to cluster the “trucks_cleaned” data frame.
# Use a random number seed of 64. Use augment to add the resulting clusters object to the the “trucks” data
# frame. Design an appropriate visualization to visualize the clusters.
set.seed(64)
kmeans_result <- kmeans(truck_cleaned, centers = 2)
truck_with_clusters <- augment(kmeans_result, truck_cleaned)
ggplot(truck_with_clusters, aes(Distance, Speeding, color = .cluster)) +
  geom_point()
