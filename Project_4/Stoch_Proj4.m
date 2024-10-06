%% Jeffrey Wong | ECE-302 | Project #4- Detection

clear
close all
clc

%% Problem 1- Radar Detection

% Part a

N = 1000; % Sample Size- Change to 1e6 to confirm probabilities are correct
A = 4;
sigma = 1;
pPresent = 0.2; % Probability of target being present
% eta = P_0/P_1 = (1-0.2)/0.2 = 0.8/0.2 = 4
Gamma = A/2 + sigma^2 * log(4)/A; % See attached file for derivations 
% Gamma acts as our parameter for decision

X = normrnd(0, sigma, 1, N);
targetPresent = (unifrnd(0,1,1,N) < pPresent);
Y = X + A.*targetPresent;
decisions = (Y > Gamma);
expErrorRate = mean(targetPresent ~= decisions);
% Probability of error = f(y > Gamma | H_0) + f(y < Gamma | H_1)
% = 1-F(Gamma | H_0) + F(Gamma | H_1)
theoreticalErrorRate = (1-pPresent)*(1-normcdf(Gamma, 0, sigma))+ pPresent*(normcdf(Gamma, A, sigma));
disp("From " + N + " samples, the experimental rate of error was " + expErrorRate);
disp("For comparison, the theoretical rate of error was " + theoreticalErrorRate);
% We don't have a lot of samples, so rates of error may differ somewhat.

% Part b

probabilities = 0.01:0.01:0.99;
[Pfs1, Pds1] = ROCSimulatorDiffMeans(A, 1, probabilities);
[Pfs2, Pds2] = ROCSimulatorDiffMeans(A, 2, probabilities);
[Pfs5, Pds5] = ROCSimulatorDiffMeans(A, 5, probabilities);
[Pfs20, Pds20] = ROCSimulatorDiffMeans(A, 20, probabilities);
figure
hold on
legend
plot(Pfs1, Pds1, 'DisplayName',"SNR = 4")
plot(Pfs2, Pds2, 'DisplayName',"SNR = 1")
plot(Pfs5, Pds5, 'DisplayName',"SNR = 0.16")
plot(Pfs20, Pds20, 'DisplayName',"SNR = 0.01")
xlabel("Pf")
ylabel("Pd")
xlim([0 1])
ylim([0 1])
title("Receiver Operating Curves for Same Mean, Different Variances")
% Part c

% See attached file for derivations
Gamma_diffcost = A/2 + log(0.4)/A; % We assume with this that sigma = 1
[PfDiffCost,PdDiffCost] = ROCSimulatorDiffMeans(A, 1, 0.4/1.4);
% If eta = 0.4, prior probability of presence is given by p/(1-p) = 0.4,
% gives p = 0.4/1.4 as solution
plot(PfDiffCost, PdDiffCost, "*", 'DisplayName', "Risk Minimization for Different Costs, SNR = 4")

% Part d

% For this part we assume A = 4 and sigma = 1 for an SNR of 4
% Cost is equal to 10*P(H_0 | 1) + P(Guess 1 | 0)

Eta = 0.1*(1-probabilities)./probabilities;
Gamma = A/2 + sigma^2*log(Eta)/A;
expectedCosts = 10 * normcdf(Gamma, A, sigma) + (1 - normcdf(Gamma, 0, sigma));
[minCost, minIndex] = min(expectedCosts);

figure
hold on
legend
title("Expected Minimized Costs versus A priori probability of presence")
plot(probabilities, expectedCosts)
plot(minIndex * 0.01, minCost, "*", 'DisplayName', "Minimum of minimized costs");
xlabel("Probability of Target Present")
ylabel("Expected cost w/ optimal decision rule")

% Part e

sigma_z = 2;

% In the same mean different variance case we make a decision based on the
% distance of our observation from the mean, hence the need for two
% different boundaries Gamma_L and Gamma_R
Gamma_dev = sqrt(abs(2*sigma^2*sigma_z^2/(sigma^2 - sigma_z^2) * log(4*(sigma/sigma_z))));
Gamma_L = A - Gamma_dev; 
Gamma_R = A + Gamma_dev;

X = normrnd(0, sigma, 1, N);
Z = normrnd(0, sigma_z, 1, N);
targetPresent = (unifrnd(0,1,1,N) < pPresent);
Y = A + X.*targetPresent + Z.*(1-targetPresent); % Essentially draws A + X if target present or A + Z if not
decisions = ((Gamma_L < Y) & (Y < Gamma_R));
expErrorRate = mean(targetPresent ~= decisions);
% Probability of error = f(y < Gamma_L | H_1) + f(Gamma_L < y < Gamma_R | H_0) + f(y > Gamma_R | H_1)
% = 1-F(Gamma_R | H_1) + F(Gamma+L | H_1) + (F(Gamma_R | H_0) - F(Gamma_L | H_0) 
theoreticalErrorRate = (pPresent)*(1-normcdf(Gamma_R, A, sigma) + normcdf(Gamma_L, A, sigma))+ (1-pPresent)*(normcdf(Gamma_R, A, sigma_z) - normcdf(Gamma_L, A, sigma_z));
disp("From " + N + " samples, the experimental rate of error was " + expErrorRate);
disp("For comparison, the theoretical rate of error was " + theoreticalErrorRate);
% Again we don't have a lot of samples, so rates of error may differ somewhat.

% ROC Plotting

[Pfs10, Pds10] = ROCSimulatorDiffVars(A, 1, 10, probabilities);
[Pfs2, Pds2] = ROCSimulatorDiffVars(A, 1, 2, probabilities);
[Pfs1p1, Pds1p1] = ROCSimulatorDiffVars(A, 1, 1.1, probabilities);
figure
hold on
legend
plot(Pfs10, Pds10, 'DisplayName',"Var_z/Var = 100")
plot(Pfs2, Pds2, 'DisplayName',"Var_z/Var = 4")
plot(Pfs1p1, Pds1p1, 'DisplayName',"Var_z/Var = 1.21")
xlabel("Pf")
ylabel("Pd")
xlim([0 1])
ylim([0 1])
title("Receiver Operating Curves for Same Mean, Different Variances")

%% Problem 2- Pattern Classification and Machine Learning

load("Iris.mat");

% We will divide the data by odd/even rows. This should be "random" enough.
trainingData = [features(1:2:end,:) labels(1:2:end)];
testingData = [features(2:2:end,:)];

% Break up data into classes, take mean and covariance
class1Data = trainingData(labels(1:2:end) == 1, 1:4);
class1mu = mean(class1Data);
class1cov = cov(class1Data);
class2Data = trainingData(labels(1:2:end) == 2, 1:4);
class2mu = mean(class2Data);
class2cov = cov(class2Data);
class3Data = trainingData(labels(1:2:end) == 3, 1:4);
class3mu = mean(class3Data);
class3cov = cov(class3Data);

% Compute likelihood sample belongs to each class
class1Likelihood = mvnpdf(testingData, class1mu, class1cov);
class2Likelihood = mvnpdf(testingData, class2mu, class2cov);
class3Likelihood = mvnpdf(testingData, class3mu, class3cov);

% Combine into one matrix for comparison
classLikelihoods = [class1Likelihood class2Likelihood class3Likelihood];
% Since the a priori probabilities for being in each class are identical
% and we assume costs of 0 for a match and 1 for a mismatch, we can use an
% ML decision rule and just select the class with the highest likelihood
% for an estimator
[~,classGuesses] = max(classLikelihoods, [], 2);
actualClasses = labels(2:2:end);
classErrorRate = mean(classGuesses ~= actualClasses);

disp("The classifier had an error rate of " + classErrorRate);

% A confusion matrix essentially says how many times the classifier said an
% object belonged to class i given it was in class j.

confMatrix = zeros(3);

% To make the confusion matrix in a simple manner we can encode the
% combined classifier guess and actual class as a single integer with guess
% + 3*actual;
for i = 1:3
    for j = 1:3
        classEncoding = ones(75,1) * (i+3*j);
        confMatrix(i,j) = sum((actualClasses * 3 + classGuesses) == classEncoding);
    end
end

disp(confMatrix)

%% Function Definitions

% Performs Receiver Operating Characteristic simulations for the different mean, same variance scenario 
function [Pfs, Pds] = ROCSimulatorDiffMeans(A, sigma, pPresents)
    N = 1000000;
    Pfs = zeros(1, length(pPresents)); % Probabilities of false alarm
    Pds = zeros(1, length(pPresents)); % Probabilities of successful detection
    for i = 1:length(pPresents)
        Eta = (1-pPresents(i))/pPresents(i);
        Gamma = A/2 + sigma^2 * log(Eta)/A;
        X = normrnd(0, sigma, 1, N);
        targetPresent = (unifrnd(0,1,1,N) < pPresents(i));
        Y = X + A.*targetPresent;
        decisions = (Y > Gamma);
        Pfs(i) = mean((decisions == 1) & (targetPresent == 0))/mean(targetPresent == 0);
        Pds(i) = mean((decisions == 1) & (targetPresent == 1))/mean(targetPresent == 1);
    end
end

% Performs Receiver Operating Characteristic simulations for the different mean, same variance scenario 
function [Pfs, Pds] = ROCSimulatorDiffVars(A, sigma, sigma_z, pPresents)
    N = 1000000;
    Pfs = zeros(1, length(pPresents)); % Probabilities of false alarm
    Pds = zeros(1, length(pPresents)); % Probabilities of successful detection
    for i = 1:length(pPresents)
        Eta = (1-pPresents(i))/pPresents(i);
        Gamma_dev = sqrt(abs(2*sigma^2*sigma_z^2/(sigma^2 - sigma_z^2) * log(Eta*(sigma/sigma_z))));
        Gamma_L = A - Gamma_dev; 
        Gamma_R = A + Gamma_dev;
        X = normrnd(0, sigma, 1, N);
        Z = normrnd(0, sigma_z, 1, N);
        targetPresent = (unifrnd(0,1,1,N) < pPresents(i));
        Y = A + X.*targetPresent + Z.*(1-targetPresent); % Essentially draws A + X if target present or A + Z if not
        decisions = ((Gamma_L < Y) & (Y < Gamma_R));
        Pfs(i) = mean((decisions == 1) & (targetPresent == 0))/mean(targetPresent == 0);
        Pds(i) = mean((decisions == 1) & (targetPresent == 1))/mean(targetPresent == 1);
    end
end