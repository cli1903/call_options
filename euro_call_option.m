%% Eurpoean Call Option Pricing

T = 1;
mu = 0.01;
n = 10^6;
K = 1.5;
sigmas = [2, 1, 0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01];

%% Naive MC

rel_errs_naive = zeros(1, length(sigmas));
est_naive = zeros(1, length(sigmas));
var_naive = zeros(1, length(sigmas));
CI_naive = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    S_T = exp(mu * T + sigma * sqrt(T) * normrnd(0, 1, 1, n));
    samples = max(S_T - K, 0);
    est = mean(samples);
    est_naive(i) = mean(samples);
    var_naive(i) = var(samples) / n;
    std_err = std(samples) / sqrt(n);
    rel_errs_naive(i) = std_err / est;
    %fprintf('naive MC est %d\n', est);
    %fprintf('naive MC relative error %d\n\n', relative_err);
    CI_naive(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end

%% Antithetic Sampling

rel_errs_anti = zeros(1, length(sigmas));
est_anti = zeros(1, length(sigmas));
var_anti = zeros(1, length(sigmas));
CI_anti = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    Z = normrnd(0, 1, 1, n);
    S_T_pos = exp(mu * T + sigma * sqrt(T) * Z);
    S_T_neg = exp(mu * T - sigma * sqrt(T) * Z);
    samples_pos = max(S_T_pos - K, 0);
    samples_neg = max(S_T_neg - K, 0);
    samples = 0.5 * (samples_pos + samples_neg);
    est = mean(samples);
    est_anti(i) = est;
    var_anti(i) = var(samples) / n;
    std_err = std(samples) / sqrt(n);
    relative_err = std_err / est;
    rel_errs_anti(i) = relative_err;
    %fprintf('European Call Option, antithetic MC est %d\n', est);
    %fprintf('European Call Option, antithetic MC relative error %d\n\n', relative_err);
    CI_anti(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end

%% Importance Sampling

rel_errs_import = zeros(1, length(sigmas));
est_import = zeros(1, length(sigmas));
var_import = zeros(1, length(sigmas));
CI_import = zeros(2, length(sigmas));

for i = 1:length(sigmas)
    sigma = sigmas(i);
    theta = (log(K) - mu * T) / (sqrt(T) * sigma);
    Z = normrnd(theta, 1, 1, n);
    S_T = exp(mu * T + sigma * sqrt(T) * Z);
    % H(z)f(z) / g(z)
    samples = max(S_T - K, 0) .* normpdf(Z, 0, 1) ./ normpdf(Z, theta, 1);
    est = mean(samples);
    est_import(i) = est;
    var_import(i) = var(samples) / n;
    std_err = std(samples) / sqrt(n);
    relative_err = std_err / est;
    rel_errs_import(i) = relative_err;
    %fprintf('importance sampling est %d\n', est);
    %fprintf('importance sampling relative error %d\n\n', relative_err);
    CI_import(:, i) = [est - 1.96 * std_err, est + 1.96 * std_err];
end

%% Plotting based on scaling factor
figure();
plot(sigmas, rel_errs_naive, sigmas, rel_errs_anti, sigmas, rel_errs_import);
legend('naive', 'antithetic', 'import sampling');
xlabel('\sigma');
ylabel('relative error');
