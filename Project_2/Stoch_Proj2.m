%% Jeffrey Wong | ECE-302 | Project #2- MMSE Estimation

clear
close all
clc

%% Problem 1 - MMSE Estimators

% Bayseian Estimator

N = 1e7; % Sample size

Y = 2.*rand(1, N) - 1; % rand() generates uniform random numbers over [0,1]
W = 4.*rand(1, N) - 2; % Added noise

X = Y + W;

YBayesEst = (0.5.*X + 0.5).*(X <= -1) + (0.5.*X - 0.5).*(X >= 1); 
% Estimator is 0.5x + 0.5 for -3<x<-1, 0.5x - 0.5 for 1<x<3, and 0 otherwise

BayesMSE = mean((Y-YBayesEst).^2); % Theoretical MSE is 1/4 using this estimator

YLinEst = 0.2.*X;
% Estimator is literally just Yest = 1/5*X
LinMSE = mean((Y-YLinEst).^2); % Theoretical MSE is 4/15 using this estimator

% Table formatting
Estimator = ["Bayesian Estimator";"Linear Estimator"];
TheoreticalMSE = [1/4; 4/15];
ExperimentalMSE = [BayesMSE; LinMSE];

MMSEEst = table(Estimator, TheoreticalMSE, ExperimentalMSE);
disp(MMSEEst);
%% Problem 2 - Noisy Estimator

sigmaY = 1;
% Three different values of sigmaR (shared for R1 and R2) to observe
% behavior at SNR >> 1, =1, and << 1.
sigmaR1 = 5;
sigmaR2 = 1;
sigmaR3 = 0.2;

[SNR1, NoisyTheoreticalMSE1, NoisyExperimentalMSE1] = noisyEstimation(sigmaY, sigmaR1);
[SNR2, NoisyTheoreticalMSE2, NoisyExperimentalMSE2] = noisyEstimation(sigmaY, sigmaR2);
[SNR3, NoisyTheoreticalMSE3, NoisyExperimentalMSE3] = noisyEstimation(sigmaY, sigmaR3);

figure
hold on
legend
semilogx([SNR3 SNR2 SNR1], [NoisyExperimentalMSE3 NoisyExperimentalMSE2 NoisyExperimentalMSE1], "DisplayName", "Experimental");
semilogx([SNR3 SNR2 SNR1], [NoisyTheoreticalMSE3 NoisyTheoreticalMSE2 NoisyTheoreticalMSE1], "DisplayName", "Theoretical");
xlabel('SNR');
ylabel('Mean Squared Error');
title('Mean Squared Error of Noisy Estimator vs SNR');

%% Problem 3 - SAT Score Analysis

load("SATs.mat");
% Why is there a random NaN in the set? Get outta here Grandma!
mathScores = SAT_Math(2:end);
verbalScores = SAT_Verbal(2:end);
totalScores = mathScores + verbalScores;
% Filtering based on total scores
mathScoresMid = mathScores(totalScores >= 1150 & totalScores <= 1250);
verbalScoresMid = verbalScores(totalScores >= 1150 & totalScores <= 1250);
mathScoresTop = mathScores(totalScores > 1320);
verbalScoresTop = verbalScores(totalScores > 1320);

SATScoreEstimator(mathScores, verbalScores, "All SAT Scores")
SATScoreEstimator(mathScoresMid, verbalScoresMid, "SAT Scores from 1150 to 1250")
SATScoreEstimator(mathScoresTop, verbalScoresTop, "SAT Scores above 1320")

% Observations- For all SAT scores there was a positive correlation
% between math and verbal scores but once we limited the scores based on
% the range we observe a negative correlation, as setting a constant total
% score would give us verbalScore = totalScore - mathScore.

%% Function Declarations

function [SNR, theoreticalMSE, experimentalMSE] = noisyEstimation(sigmaY, sigmaR)
    N = 1e7;
    Y = normrnd(1, sigmaY, 1, N);
    R1 = normrnd(0, sigmaR, 1, N);
    R2 = normrnd(0, sigmaR, 1, N);
    X1 = Y + R1;
    X2 = Y + R2;
    YNoisyEst = (sigmaR^2 + sigmaY^2.*(X1+X2))/(2*sigmaY^2 + sigmaR^2);
    experimentalMSE = mean((Y-YNoisyEst).^2);
    theoreticalMSE = (sigmaR*sigmaY)^2/(2*sigmaY^2 + sigmaR^2);
    SNR = (sigmaY/sigmaR)^2;
end

function SATScoreEstimator(mathScores, verbalScores, datasetName)
    muM = mean(mathScores);
    muV = mean(verbalScores);
    sigmaM = std(mathScores);
    sigmaV = std(verbalScores);
    covMV = cov(mathScores,verbalScores);
    rho = covMV(1,2) / (sigmaM*sigmaV); %covMV returns a 2x2 matrix, we want the cross covariance term
    
    range = linspace(200,800,1e4);
    Vest = muV + (range-muM)*(rho*sigmaV/sigmaM);
    
    figure
    hold on
    scatter(mathScores, verbalScores)
    plot(linspace(200,800,1e4),Vest)
    xlabel("SAT Math Score")
    ylabel("SAT Verbal Score")
    title("Scores and Estimation for " + datasetName)
end