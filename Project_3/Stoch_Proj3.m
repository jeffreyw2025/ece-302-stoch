%% Jeffrey Wong | ECE-302 | Project #3- ML Estimation

clear
close all
clc

%% Problem 1 - Random Draws of Variables

% Note: See attached file for derivation of estimators.

N = 250;

% Exponential RVs

[exp_experimentalMSE_half, exp_bias_half, exp_variance_half] = expSample(N, 0.5);
[exp_experimentalMSE_2, exp_bias_2, exp_variance_2] = expSample(N, 2);
[exp_experimentalMSE_5, exp_bias_5, exp_variance_5] = expSample(N, 5);

figure
title("Exponential Draw Analysis")
subplot(3,1,1)
% If we take only 1 sample MMSE grows untenably large
semilogx(2:N, exp_experimentalMSE_half(2:N), 'DisplayName',"\lambda = 2")
hold on % Need hold on here so that plots correctly render logarithmic x scale
semilogx(2:N, exp_experimentalMSE_2(2:N), 'DisplayName',"\lambda = 0.5")
semilogx(2:N, exp_experimentalMSE_5(2:N), 'DisplayName',"\lambda = 0.2")
legend
title("Experimental MSE for Exponential Draws")
xlabel("Number of Samples")
ylabel("MSE")

subplot(3,1,2)
legend
semilogx(1:N, exp_variance_half, 'DisplayName',"\lambda = 2")
hold on
semilogx(1:N, exp_variance_2, 'DisplayName',"\lambda = 0.5")
semilogx(1:N, exp_variance_5, 'DisplayName',"\lambda = 0.2")
legend
title("Variance for Exponential Draws")
xlabel("Number of Samples")
ylabel("Variance")

subplot(3,1,3)
legend
semilogx(1:N, exp_bias_half, 'DisplayName',"\lambda = 2")
hold on
semilogx(1:N, exp_bias_2, 'DisplayName',"\lambda = 0.5")
semilogx(1:N, exp_bias_5, 'DisplayName',"\lambda = 0.2")
legend
title("Estimator Bias for Exponential Draws")
xlabel("Number of Samples")
ylabel("Bias")

% Rayleigh RVs

[ray_experimentalMSE_half, ray_bias_half, ray_variance_half] = raySample(N, 0.5);
[ray_experimentalMSE_2, ray_bias_2, ray_variance_2] = raySample(N, 2);
[ray_experimentalMSE_5, ray_bias_5, ray_variance_5] = raySample(N, 5);

figure
title("Rayleigh Draw Analysis")
subplot(3,1,1)
semilogx(1:N, ray_experimentalMSE_half, 'DisplayName',"\alpha = 0.5")
hold on
semilogx(1:N, ray_experimentalMSE_2, 'DisplayName',"\alpha = 2")
semilogx(1:N, ray_experimentalMSE_5, 'DisplayName',"\alpha = 5")
legend
title("Experimental MSE for Rayleigh Draws")
xlabel("Number of Samples")
ylabel("MSE")

subplot(3,1,2)
legend
semilogx(1:N, ray_variance_half, 'DisplayName',"\alpha = 0.5")
hold on
semilogx(1:N, ray_variance_2, 'DisplayName',"\alpha = 2")
semilogx(1:N, ray_variance_5, 'DisplayName',"\alpha = 5")
legend
title("Variance for Rayleigh Draws")
xlabel("Number of Samples")
ylabel("Variance")

subplot(3,1,3)
legend
semilogx(1:N, ray_bias_half, 'DisplayName',"\alpha = 0.5")
hold on
semilogx(1:N, ray_bias_2, 'DisplayName',"\alpha = 2")
semilogx(1:N, ray_bias_5, 'DisplayName',"\alpha = 5")
legend
title("Estimator Bias for Rayleigh Draws")
xlabel("Number of Samples")
ylabel("Bias")

%% Problem 2 - Guess the Distribution!

load("data.mat")
n_data = length(data);

% To find the max likelihood for each possible distribution, we take our
% estimator for that distribution and evaluate the (log) likelihood function
% over the data set with parameter equal to the estimator. The function
% with the greater (log) likelihood is more likely to have been drawn from.

% Exponential RV Analysis

% Our ML Estimator if data is exponential occurs at sample mean

data_exp_estimator = 1/mean(data);
disp("If the data is exponentially distributed, it has parameter lambda = " + data_exp_estimator)
log_exp_likelihood = n_data * log(data_exp_estimator) - data_exp_estimator * sum(data);

% Our ML Estimator if data is Rayleigh occurs at the square root of the
% mean of values squared

data_ray_estimator = sqrt(sum(data.^2)/(2*n_data));
disp("If the data is Rayleigh distributed, it has parameter alpha = " + data_ray_estimator)

log_ray_likelihood = sum(log(data)) - (2 * n_data * log(data_ray_estimator)) - sum(data.^2)/(2*data_ray_estimator^2);

if log_exp_likelihood > log_ray_likelihood
    disp("It is more likely that the data was drawn from an exponential distribution.")
else
    disp("It is more likely that the data was drawn from an Rayleigh distribution.")
end

%% Function Declarations

function [experimentalMSE, bias, variance] = expSample(numSamples, mu)
    trials = 1e4;
    experimentalMSE = zeros(1, numSamples);
    bias = zeros(1, numSamples);
    variance = zeros(1, numSamples);
    expDraw = exprnd(mu, trials, numSamples);
    for i = 1:numSamples
        subDraw = expDraw(:,1:i);
        expEstimator = 1./mean(subDraw, 2); % We will estimate the mean of the exponential as the mean of our observations per trial
        % Note that our actual parameter lambda is equal to 1/mu.
        experimentalMSE(i) = mean((1/mu - expEstimator).^2);
        variance(i) = mean((mu - subDraw).^2, "all");
        bias(i) = mean(1/mu - expEstimator);
    end
end

function [experimentalMSE, bias, variance] = raySample(numSamples, b)
    trials = 1e4;
    experimentalMSE = zeros(1, numSamples);
    bias = zeros(1, numSamples);
    variance = zeros(1, numSamples);
    rayDraw = raylrnd(b, trials, numSamples);
    for i = 1:numSamples
        subDraw = rayDraw(:,1:i);
        rayEstimator = sqrt(mean(subDraw.^2, 2)/2); % We will estimate the mean of the exponential as the mean of our observations per trial
        experimentalMSE(i) = mean((b - rayEstimator).^2);
        variance(i) = mean((b - subDraw).^2, "all");
        bias(i) = mean(b - rayEstimator);
    end
end