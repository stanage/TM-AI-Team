close all; clc; clear;

%--------LOAD DATASET
dataset = load('dataset.csv');

%--------SHUFFLE ROWS IN DATASET
size_dataset = size(dataset);
m_dataset = size_dataset(1);
idx = randperm(m_dataset);
rand_dataset = dataset;
rand_dataset(idx, :) = dataset(:, :);
old_dataset = dataset;
dataset = rand_dataset;

%--------SPLIT DATA INTO FEATURES AND TARGET
X_data = dataset(:, 1:4);
y_data = dataset(:, 5);

%--------FEATURE NORMALIZATION
t = ones(length(X_data), 1);
X_norm = (X_data - (t * mean(X_data))) ./ (t * std(X_data));
y_log = log(1+y_data);
X = X_norm;
y = y_data;

%--------SPLIT DATA INTO TRAINING AND TEST SETS
data = [X, y];
train_data = data(1:30, :);
test_data = data(31:42, :);

[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM(train_data, test_data, 0, 15, 'sig')




