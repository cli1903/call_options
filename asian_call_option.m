%% Asian Call Option Pricing

T = 1;
mu = 0.01;
n = 10^6;
k = 5;
K = 1.5;
sigmas = [2, 1, 0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01];

%% Naive MC

rel_errs_naive = zeros(1, length(sigmas));
est_naive = zeros(1, length(sigmas));
CI_naive = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    S_t = exp_brownian_motion(T, mu, sigma, n, k, 0, 0);
    avg_S_t = mean(S_t, 2);
    samples = max(avg_S_t - K, 0);
    est = mean(samples);
    est_naive(i) = est;
    std_err = std(samples) / sqrt(n);
    relative_err = std_err / est;
    rel_errs_naive(i) = relative_err;
    %fprintf('naive MC est %d\n', est);
    %fprintf('naive MC relative error %d\n\n', relative_err);
    CI_naive(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end

%% Antithetic Sampling

rel_errs_anti = zeros(1, length(sigmas));
est_anti = zeros(1, length(sigmas));
CI_anti = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    S_t = exp_brownian_motion(T, mu, sigma, n, k, 1, 0);
    S_t_pos = S_t(:, :, 1);
    S_t_neg = S_t(:, :, 2);
    avg_S_t_pos = mean(S_t_pos, 2);
    avg_S_t_neg = mean(S_t_neg, 2);
    samples_pos = max(avg_S_t_pos - K, 0);
    samples_neg = max(avg_S_t_neg - K, 0);
    samples = 0.5 * (samples_pos + samples_neg);
    est = mean(samples);
    est_anti(i) = est;
    std_err = std(samples) / sqrt(n);
    relative_err = std_err / est;
    rel_errs_anti(i) = relative_err;
    %fprintf('antithetic sampling est %d\n', est);
    %fprintf('antithetic sampling relative error %d\n\n', relative_err);
    CI_anti(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end

%% Importance Sampling

%plot_conditional((log(K) - mu * T/k) / (sqrt(T/K) * 1));
%plot_conditional((-mu * T/k) / (sqrt(T/K) * 0.01));

rel_errs_import = zeros(1, length(sigmas));
est_import = zeros(1, length(sigmas));
CI_import = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    [S_t, Z, thetas] = exp_brownian_motion_import(T, mu, sigma, n, k, K);
    avg_S_t = mean(S_t, 2);
    samples = max(avg_S_t - K, 0) .* prod(normpdf(Z, 0, 1), 2) ./ prod(normpdf(Z, thetas, 1), 2);
    est = mean(samples);
    est_import(i) = est;
    std_err = std(samples) / sqrt(n);
    relative_err = std_err / est;
    rel_errs_import(i) = relative_err;
    %fprintf('importance sampling est %d\n', est);
    %fprintf('importance sampling relative error %d\n\n', relative_err);
    CI_import(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end


%% Functions
function S_t = exp_brownian_motion(T, mu, sigma, n, k, antithetic, theta)
if antithetic
    S_t = zeros(n, k, 2);
else
    S_t = zeros(n, k, 1);
end
dt = T/k;
Z = zeros(n, k);
Z(:, 1) = normrnd(theta, 1, n, 1);
S_t(:, 1, 1) = exp(mu * dt + sigma * sqrt(dt) * Z(:, 1));
if antithetic
    S_t(:, 1, 2) = exp(mu * dt - sigma * sqrt(dt) * Z(:, 1));
end
for i = 2:k
    Z(:, i) = normrnd(theta, 1, n, 1);
    S_t(:, i, 1) = S_t(:, i-1, 1) .* exp(mu * dt + sigma * sqrt(dt) * Z(:, i));
    if antithetic
        S_t(:, i, 2) = S_t(:, i-1, 2) .* exp(mu * dt - sigma * sqrt(dt) * Z(:, i));
    end
end
end

function [S_t, Z, thetas] = exp_brownian_motion_import(T, mu, sigma, n, k, K)
S_t = zeros(n, k);
dt = T/k;
thetas = zeros(1,k);%(-mu * dt) / (sqrt(dt) * sigma) * ones(1, k);
thetas(1, 1) = (log(K) - mu * dt) / (sqrt(dt) * sigma);
Z = zeros(n, k);
Z(:, 1) = normrnd(thetas(1, 1), 1, n, 1);
S_t(:, 1) = exp(mu * dt + sigma * sqrt(dt) * Z(:, 1));
for i = 2:k
    Z(:, i) = normrnd(thetas(1, i), 1, n, 1);
    S_t(:, i) = S_t(:, i-1, 1) .* exp(mu * dt + sigma * sqrt(dt) * Z(:, i));
end
end

function plot_conditional(non_zero_thresh)
figure();
fplot(@(x) normpdf(x) .* (x > non_zero_thresh), [-10, 10]);
end