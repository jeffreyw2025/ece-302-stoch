%% Jeffrey Wong | ECE-302 | Project #1- Dungeons and Dragons

clear
close all
clc

%% Problem 1- Character Creation

N = 10000000; % We will consistently use 10000000 trials in each sample
disp("With " + N + " trials...");

% Part a

% Expected probability of an 18 is (1/6)^3 = 1/216, or about 0.463%

basicStatGen = sum(randi(6, 3, N)); % randi generates a 3x10000 matrix, sum will take the sum of each of 3 rows to simulate 3d6
prob18basic = mean(basicStatGen == 18);
disp("The basic method generated a perfect score with probability " + prob18basic);

% Part b

% Expected probability of an 18 with the fun method is 1 - (215/216)^3, or about 1.38%

funStatGen = sum(randi(6, 3, 3, N)); % randi generates a 3x3x1000000 matrix, 3 attempts at 3d6
funStatFinal = max(funStatGen); % max takes the best of these attempts
figure
histogram(funStatFinal(1,1,:),2.5:1:18.5);
title("Experimental PDF of Ability Scores generated using Fun Method");
xlabel("Score")
ylabel("Absolute Frequency over " + N + " trials");
prob18fun = mean(funStatFinal == 18);
disp("The fun method generated a perfect score with probability " + prob18fun);

% Part c

% Expected probability of all 18s with the fun method is (1 - (215/216)^3)^6,
% or about 1 in 1.4*10^11 odds

multiFunStatGen = max(sum(randi(6, 3, 3, 6, 100)));
perfectAbility = (multiFunStatGen == 18);
allPerfectAbilities = (sum(perfectAbility)) > 5;
probAllPerfectAbilities = mean(allPerfectAbilities); % Probably generates to 0 due to extreme odds
disp("The fun method generated all perfect abilities with probability " + probAllPerfectAbilities);

% Part d

% Rolling an 8 or less on a 3d6 has a probability of (1+3+6+10+15+21)/216 =
% 56/216 = 7/27.
% Getting a 9 on any one ability using the fun method has a probability of
% [(3C1)(25/216)(7/27)^2 + (3C2)(25/216)^2(7/27) + (3C3)(25/216)^3] = 0.0353
% Thus getting a 9 on all six abilities using the fun method has a
% probability of (0.0353)^6 or about 1 in 516 million odds

averageAbility = (multiFunStatGen == 9);
allAverageAbilities = (sum(averageAbility)) > 5;
probAllAverageAbilities = mean(allAverageAbilities); % Also probably generates to 0 due to extreme odds
disp("The fun method generated ability scores of all 9 with probability " + probAllAverageAbilities);


%% Problem 2- Combat Encounter

% Part a

% The trolls have an average of 10/4 = 2.5 HP (which is pretty sad for a troll)
% and Fireball does an average of 1.5*2 = 3 damage with a probability .25 of doing >3 damage

trollHP = randi(4, 1, 6, N); % The extra 1 is needed to add an extra dimension to make a later logical array work
meanTrollHP = mean(trollHP, "all");
disp("Trolls had on average " + meanTrollHP + " HP");


fireballDamage = sum(randi(2,2,6,N));
meanFireballDamage = mean(fireballDamage, "all");
probFireballDamage4 = mean(fireballDamage > 3, "all");
disp("Fireball on average " + meanFireballDamage + " damage and inflicted 4 damage with probability " + probFireballDamage4);


% Part b

% The pmf for Troll HP, represented by discrete random variable H, is given by 
% P(H = 1) = P(H = 2) = P(H = 3) = P(H = 4) = 0.25
% The pmf for Fireball damage, represented by variable D, is P(D = 2) =
% 0.25, P(D = 3) = 0.5, and P(D = 4) = 0.25
% See figures 1 and 2 for a (representation) of the pdf

figure
bar(1:4, [0.25 0.25 0.25 0.25]);
title("PDF of Troll HP");
xlabel("HP")
ylabel("Outcome Probability");

figure
bar(2:4, [0.25 0.5 0.25]);
title("PDF of Fireball Damage");
xlabel("Damage");
ylabel("Outcome Probability");

% Part c

% The probability of any one troll surviving is 0.25(0.25) + 0.25(0.75) or
% 25%, meaning the probability all six troll die is (0.75)^6 or about .178.

trollDead = (trollHP <= fireballDamage);
allTrollsDead = (sum(trollDead) == 6);
probAllTrollsDie = mean(allTrollsDead);
disp("Fireball killed all six trolls with probability " + probAllTrollsDie);

% Part d

% A troll survives on 3 HP vs 2 damage (prob 1/16), 4 HP vs 2 damage (1/16), 
% or 4 HP vs 3 damage (1/8), and survives 25% of the time, giving an expected
% HP on survival of (1/16 + 1/16 + 1/8 * 2)/(1/4) = 1.25

leftoverHP = (trollHP - fireballDamage) .* ~trollDead;
avgLeftoverHP = sum(leftoverHP,"all")/sum(~trollDead,"all");
disp("Given a troll survived, it would have on average " + avgLeftoverHP + " HP left.");

% Part e

% The expected damage is given by 0.5(7) for the sword + (0.5)^2(2.5) for
% the hammer or 4.125 total

swordDamage = sum(randi(6, 2, N));
hammerDamage = randi(4, 1, N);
swordHit = (randi(20, 1, N) > 10);
hammerHit = (randi(20, 1, N) > 10).*swordHit; % Hammer can only hit if sword hits
totalDamage = swordDamage.*swordHit + hammerDamage.*hammerHit;
meanShedjamDamage = mean(totalDamage);

disp("Shedjam inflicted an average of " + meanShedjamDamage + " damage");
