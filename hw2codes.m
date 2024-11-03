%q1
% Define the mean vector and covariance matrix
mu1 = [-0.75; 0.5];
Sigma1 = [0.5, 0.3; 0.3, 0.8];

% Set the random seed for reproducibility
rng('default') ;

% Generate 10 samples
n_samples_10 = 10;
samples_10 = mvnrnd(mu1, Sigma1, n_samples_10);

% Plot the 10 samples
figure;
scatter(samples_10(:,1), samples_10(:,2), 'filled');
title('10 Samples from Multivariate Normal Distribution');
xlabel('X1');
ylabel('X2');
grid on;

% Estimate ML parameters for 10 samples
mu_10_ML = mean(samples_10)';
Sigma_10_ML = cov(samples_10)*9/10;

% Display the estimated parameters for 10 samples
disp('Estimated Mean (10 samples):');
disp(mu_10_ML);
disp('Estimated Covariance Matrix (10 samples):');
disp(Sigma_10_ML);

% Generate 1000 samples
n_samples_1000 = 1000;
samples_1000 = mvnrnd(mu1, Sigma1, n_samples_1000);

% Estimate ML parameters for 1000 samples
mu_1000_ML = mean(samples_1000)';
Sigma_1000_ML = cov(samples_1000)*999/1000;

% Display the estimated parameters for 1000 samples
disp('Estimated Mean (1000 samples):');
disp(mu_1000_ML);
disp('Estimated Covariance Matrix (1000 samples):');
disp(Sigma_1000_ML);

% Plot the 1000 samples
figure;
scatter(samples_1000(:,1), samples_1000(:,2), 'filled');
title('1000 Samples from Multivariate Normal Distribution');
xlabel('X1');
ylabel('X2');
grid on;

%Q2
clear all
rng('default')
% Parameters
true_mu = 3;      % Given mean of the distribution
sigma = 0.7;      % Known fixed standard deviation
prior_mu = 2.8;   % Prior mean for Bayesian estimation
prior_var = 0.8;  % Prior variance for Bayesian estimation

% Part (i): Generate and plot 25 samples, find ML estimate
n_samples_25 = 25;

samples_25 = true_mu + (sigma) * randn(n_samples_25, 1);

% ML estimate for 25 samples
ml_estimate_25 = mean(samples_25);

% Plot the 25 samples
figure;
plot(samples_25, 'bo');
hold on;
% yline(ml_estimate_25, 'r', 'LineWidth', 1.5);
yline(true_mu, 'g--', 'LineWidth', 1.5);
title('25 Samples for Normal Distribution');
legend('Samples', 'True Mean');
xlabel('Sample Index');
ylabel('Value');
hold off;

w1_ii       = (n_samples_25*prior_var )/(n_samples_25*prior_var  + sigma );
w2_ii       = (sigma )/(n_samples_25*prior_var  + sigma );

mu_map_ii   = w1_ii * ml_estimate_25 + w2_ii * prior_mu;
    
disp('--- Results for 25 samples ---');
disp(['ML Estimate of mu: ', num2str(ml_estimate_25)]);
disp(['MAP Estimate of mu: ', num2str(mu_map_ii)]);
rng(41)
% Part (iii): Repeat with 1000 samples
n_samples_1000 = 1000;
samples_1000 = true_mu + sigma * randn(n_samples_1000, 1);

% ML estimate for 1000 samples
ml_estimate_1000 = mean(samples_1000);

% MAP estimate for 1000 samples
map_estimate_1000 = (prior_var * ml_estimate_1000 + sigma^2 * prior_mu) / (sigma^2 + prior_var);

disp('--- Results for 1000 samples ---');
disp(['ML Estimate of mu: ', num2str(ml_estimate_1000)]);
disp(['MAP Estimate of mu: ', num2str(map_estimate_1000)]);

% Plot the 1000 samples
figure;
plot(samples_1000, 'bo', 'MarkerSize', 3);
hold on;
% yline(ml_estimate_1000, 'r', 'LineWidth', 1.5);
yline(true_mu, 'g--', 'LineWidth', 1.5);
title('1000 Samples for Normal Distribution');
legend('Samples',  'True Mean');
xlabel('Sample Index');
ylabel('Value');
hold off;



% Q3 - Minimum Error-Rate Classifier with Distinct Style
clear; clc; close all;


% Define means and covariance matrix for two classes
mean_A = [-1; -1];
mean_B = [1; 1];
covariance = [1.4, 0.2; 0.2, 0.28];

% Generate 500 samples for each class
data_A = mvnrnd(mean_A, covariance, 500);
data_B = mvnrnd(mean_B, covariance, 500);

% Randomly split data into training (250) and testing (250)
train_indices_A = randperm(500, 250);
train_indices_B = randperm(500, 250);
A_train = data_A(train_indices_A, :);
A_test = data_A(setdiff(1:500, train_indices_A), :);
B_train = data_B(train_indices_B, :);
B_test = data_B(setdiff(1:500, train_indices_B), :);

%% Plot 1: Training Samples (No Boundary)
figure;
scatter(A_train(:,1), A_train(:,2), 30, 'm', 'filled'); hold on;
scatter(B_train(:,1), B_train(:,2), 30, 'c', 'filled');
title('Training Set - Class A and Class B');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Training', 'Class B Training');
grid on; hold off;

%% Plot 2: Testing Samples (No Boundary)
figure;
scatter(A_test(:,1), A_test(:,2), 30, 'm', 'filled'); hold on;
scatter(B_test(:,1), B_test(:,2), 30, 'c', 'filled');
title('Testing Set - Class A and Class B');
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Testing', 'Class B Testing');
grid on; hold off;

%% Analytical Decision Boundary
normal_vector = (mean_A - mean_B)';
midpoint = (mean_A + mean_B) / 2;
boundary_function = @(x, y) normal_vector * inv(covariance) * ([x; y] - midpoint);

% Classification and Error Calculation for Training Set
train_set = [A_train; B_train];
train_labels = [ones(250, 1); 2 * ones(250, 1)];
predicted_train_labels = arrayfun(@(i) classify_point(train_set(i, :)', boundary_function), 1:length(train_set))';
training_error = mean(predicted_train_labels ~= train_labels) * 100;

% Classification and Error Calculation for Testing Set
test_set = [A_test; B_test];
test_labels = [ones(250, 1); 2 * ones(250, 1)];
predicted_test_labels = arrayfun(@(i) classify_point(test_set(i, :)', boundary_function), 1:length(test_set))';
testing_error = mean(predicted_test_labels ~= test_labels) * 100;

%% Plot 3: Training Data with Analytical Boundary
figure;
scatter(A_train(:, 1), A_train(:, 2), 30, 'm', 'filled'); hold on;
scatter(B_train(:, 1), B_train(:, 2), 30, 'c', 'filled');
fimplicit(@(x, y) boundary_function(x, y), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
title(sprintf('Analytical Boundary on Training Set (Error: %.2f%%)', training_error));
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Train', 'Class B Train', 'Boundary');
grid on; hold off;

%% Plot 4: Testing Data with Analytical Boundary
figure;
scatter(A_test(:, 1), A_test(:, 2), 30, 'm', 'filled'); hold on;
scatter(B_test(:, 1), B_test(:, 2), 30, 'c', 'filled');
fimplicit(@(x, y) boundary_function(x, y), [-4 4 -4 4], 'k', 'LineWidth', 1.5);
title(sprintf('Analytical Boundary on Testing Set (Error: %.2f%%)', testing_error));
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Test', 'Class B Test', 'Boundary');
grid on; hold off;

%% Maximum Likelihood Estimation (MLE) for Decision Boundary
mean_A_ML = mean(A_train)';
mean_B_ML = mean(B_train)';
cov_ML = cov([A_train; B_train]);

normal_vector_ML = (mean_A_ML - mean_B_ML)';
midpoint_ML = (mean_A_ML + mean_B_ML) / 2;
boundary_function_ML = @(x, y) normal_vector_ML * inv(cov_ML) * ([x; y] - midpoint_ML);

% MLE Classification and Error Calculation for Training Set
predicted_train_labels_ML = arrayfun(@(i) classify_point(train_set(i, :)', boundary_function_ML), 1:length(train_set))';
training_error_ML = mean(predicted_train_labels_ML ~= train_labels) * 100;

% MLE Classification and Error Calculation for Testing Set
predicted_test_labels_ML = arrayfun(@(i) classify_point(test_set(i, :)', boundary_function_ML), 1:length(test_set))';
testing_error_ML = mean(predicted_test_labels_ML ~= test_labels) * 100;

%% Plot 5: Training Data with MLE Boundary
figure;
scatter(A_train(:, 1), A_train(:, 2), 30, 'm', 'filled'); hold on;
scatter(B_train(:, 1), B_train(:, 2), 30, 'c', 'filled');
fimplicit(@(x, y) boundary_function_ML(x, y), [-4 4 -4 4], 'g', 'LineWidth', 1.5);
title(sprintf('MLE Boundary on Training Set (Error: %.2f%%)', training_error_ML));
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Train', 'Class B Train', 'MLE Boundary');
grid on; hold off;

%% Plot 6: Testing Data with MLE Boundary
figure;
scatter(A_test(:, 1), A_test(:, 2), 30, 'm', 'filled'); hold on;
scatter(B_test(:, 1), B_test(:, 2), 30, 'c', 'filled');
fimplicit(@(x, y) boundary_function_ML(x, y), [-4 4 -4 4], 'g', 'LineWidth', 1.5);
title(sprintf('MLE Boundary on Testing Set (Error: %.2f%%)', testing_error_ML));
xlabel('Feature 1'); ylabel('Feature 2');
legend('Class A Test', 'Class B Test', 'MLE Boundary');
grid on; hold off;

%% Helper function for classification

% Q4 - Non-parametric Density Estimation with Parzen Window (Gaussian and Epanechnikov)
clear; clc; close all;

%% Step 1: Define the Mean and Covariance Matrix
mean_vector = [-1.5; 1.5];
cov_matrix = [0.8, 0.2; 0.2, 0.6];

%% Step 2: Plot the Given Normal Distribution in 3D
[x, y] = meshgrid(linspace(-5, 5, 500), linspace(-5, 5, 500));
grid_points = [x(:), y(:)];
original_distribution = mvnpdf(grid_points, mean_vector', cov_matrix);
original_distribution = reshape(original_distribution, size(x));

figure;
surf(x, y, original_distribution, 'EdgeColor', 'none');
title('Original Gaussian Distribution');
xlabel('x_1'); ylabel('x_2'); zlabel('Density');
colormap jet;
colorbar;
view(3);

%% Step 3: Generate 50 Samples from the Given Distribution
num_samples = 5000;
data_samples = mvnrnd(mean_vector, cov_matrix, num_samples);

% Plot the generated samples
figure;
scatter3(data_samples(:, 1), data_samples(:, 2), zeros(num_samples, 1), 50, 'filled');
title('Generated Samples');
xlabel('x_1'); ylabel('x_2'); zlabel('Density');
view(3);

%% Step 4: Parzen Window Density Estimation (Gaussian Kernel) with Varying h1
% Define grid for density estimation
[x_grid, y_grid] = meshgrid(linspace(-5, 5, 500), linspace(-5, 5, 500));
grid_points = [x_grid(:), y_grid(:)];

% Different initial volume parameters h1 for experimentation
h1_values = linspace(0.5, 2.5, 9);  % Nine values of h1

figure;
tiledlayout(3, 3);
for i = 1:length(h1_values)
    % Calculate initial volume V0 and volume Vn
    h1 = h1_values(i);
    V0 = h1^3;
    Vn = V0 / sqrt(num_samples);   % Volume shrinking formula
    hn = Vn^(1/3);                 % Bandwidth based on volume

    % Perform Parzen Window Density Estimation using Gaussian Kernel
    density_estimation = zeros(size(grid_points, 1), 1);
    for j = 1:num_samples
        density_estimation = density_estimation + mvnpdf(grid_points, data_samples(j, :), hn^2 * eye(2));
    end
    density_estimation = density_estimation / num_samples;  % Normalize by number of samples
    density_estimation = reshape(density_estimation, size(x_grid));

    % Plot the estimated density for current h1
    nexttile;
    surf(x_grid, y_grid, density_estimation, 'EdgeColor', 'none');
    title(sprintf('Gaussian Kernel (h1 = %.1f)', h1));
    xlabel('x_1'); ylabel('x_2'); zlabel('Estimated Density');
    colormap turbo;
    colorbar;
    view(3);
end

%% Step 5: Parzen Window Density Estimation (Epanechnikov Kernel) with Varying h1
% Epanechnikov kernel function
epanechnikov_kernel = @(u) 0.75 * (1 - u.^2) .* (abs(u) <= 1);

figure;
tiledlayout(3, 3);
for i = 1:length(h1_values)
    % Calculate initial volume V0 and volume Vn
    h1 = h1_values(i);
    V0 = h1^3;
    Vn = V0 / sqrt(num_samples);   % Volume shrinking formula
    hn = Vn^(1/3);                 % Bandwidth based on volume

    % Perform Parzen Window Density Estimation using Epanechnikov Kernel
    density_estimation = zeros(size(grid_points, 1), 1);
    for j = 1:num_samples
        diff = (grid_points - data_samples(j, :)) / hn;
        distance_sq = sum(diff.^2, 2);
        density_estimation = density_estimation + epanechnikov_kernel(distance_sq);
    end
    density_estimation = density_estimation / (num_samples * Vn);  % Normalize by number of samples and volume
    density_estimation = reshape(density_estimation, size(x_grid));

    % Plot the estimated density for current h1
    nexttile;
    surf(x_grid, y_grid, density_estimation, 'EdgeColor', 'none');
    title(sprintf('Epanechnikov Kernel (h1 = %.1f)', h1));
    xlabel('x_1'); ylabel('x_2'); zlabel('Estimated Density');
    colormap turbo;
    colorbar;
    view(3);
end

% Improved PCA Analysis on Fisher Iris Dataset for Different Feature Pairs with Concise Titles
clear; clc; close all;

% Load the Fisher Iris Dataset
data = load('fisheriris');
features = data.meas;
classes = data.species;

% Define feature sets
feature_sets = {[1, 2], [3, 4]};
feature_names = {'Sepal Dimensions', 'Petal Dimensions'};

for k = 1:length(feature_sets)
    X = features(:, feature_sets{k});
    feature_title = feature_names{k};

    % Plot the original 2D scatter plot of the data
    figure;
    gscatter(X(:, 1), X(:, 2), classes, 'rgb', 'osd');
    xlabel('Feature 1');
    ylabel('Feature 2');
    title(['Scatter Plot: ' feature_title]);
    grid on;

    % Perform PCA on the selected features
    [coeff, score, ~] = pca(X);

    % First and second principal components
    basis = coeff(:, 1);   % Most principal direction (PC1)
    normal = coeff(:, 2);  % Least principal direction (PC2)
    meanX = mean(X, 1);    % Mean of the data

    % Calculate fitted points (projections onto the principal component)
    [n, p] = size(X);
    Xfit = repmat(meanX, n, 1) + score(:, 1) * coeff(:, 1)'; % Projection onto PC1

    % Plot the fitted PCA line and projections for each class
    figure;
    gscatter(X(:, 1), X(:, 2), classes, 'rgb', 'osd'); hold on;
    dirVect = coeff(:, 1);  % Direction vector for PC1
    t = [min(score(:, 1)) - 0.2, max(score(:, 1)) + 0.2];
    endpts = [meanX + t(1) * dirVect'; meanX + t(2) * dirVect'];
    plot(endpts(:, 1), endpts(:, 2), 'k-', 'LineWidth', 1.5);

    % Add projections onto the PCA line for each class
    for i = 1:n
        projPoint = meanX + score(i, 1) * coeff(:, 1)';
        plot([X(i, 1), projPoint(1)], [X(i, 2), projPoint(2)], 'k--');
    end

    legend('Setosa', 'Versicolor', 'Virginica', 'PC Line', 'Location', 'southeast');
    title(['PCA Fit with Projections (' feature_title ')']);
    xlabel('Feature 1');
    ylabel('Feature 2');
    grid on;
    hold off;

    % Separate data by class for histogram plotting on PC1 and PC2
    class1_PC1 = score(strcmp(classes, 'setosa'), 1);      % Projection onto PC1 for Setosa
    class2_PC1 = score(strcmp(classes, 'versicolor'), 1);  % Projection onto PC1 for Versicolor
    class3_PC1 = score(strcmp(classes, 'virginica'), 1);   % Projection onto PC1 for Virginica

    class1_PC2 = score(strcmp(classes, 'setosa'), 2);      % Projection onto PC2 for Setosa
    class2_PC2 = score(strcmp(classes, 'versicolor'), 2);  % Projection onto PC2 for Versicolor
    class3_PC2 = score(strcmp(classes, 'virginica'), 2);   % Projection onto PC2 for Virginica

    % Plot histograms for each class on the most principal direction (PC1)
    figure;
    hold on;
    histogram(class1_PC1, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(class2_PC1, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(class3_PC1, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    xlabel('PC1 Projection');
    ylabel('Frequency');
    title(['Class Distribution on PC1 (' feature_title ')']);
    legend('Setosa', 'Versicolor', 'Virginica');
    hold off;

    % Plot histograms for each class on the least principal direction (PC2)
    figure;
    hold on;
    histogram(class1_PC2, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(class2_PC2, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    histogram(class3_PC2, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    xlabel('PC2 Projection');
    ylabel('Frequency');
    title(['Class Distribution on PC2 (' feature_title ')']);
    legend('Setosa', 'Versicolor', 'Virginica');
    hold off;
end


function label = classify_point(point, decision_function)
    if decision_function(point(1), point(2)) > 0
        label = 1;
    else
        label = 2;
    end
end

function label = classify_data(x, decision_func)
    if decision_func(x(1), x(2)) > 0
        label = 1;
    else
        label = 2;
    end
end



