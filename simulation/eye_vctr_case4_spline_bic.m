% use BIC to search best n_knots and q for spline method 

clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. generate simulation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% varying coefficient index
n = 2000;              % sample size
t = rand(n,1);         % index variable
[t, ~] = sort(t);

% 2D varying coefficient
R = 20;
S = 64;
sp = 10;
bR = (1:R)' / R;
bS = (1:S)' / S;

At = zeros(n,R,S);
for r = 1:sp
    for s = 1:sp
        if s < r
            At(:,r,s) = sin(2*pi*(t-0.5)) * sqrt(bR(r)) * sqrt(bS(s));
        else
            At(:,r,s) = sqrt(bR(r)) * sqrt(bS(s));
        end
    end
end

% True coefficients for regular one-way covariates
p0 = 5;
beta = [1,1,0,0,0]';

% construct X tensor variates and Z one-way covariates
X = randn(n,R,S);      % n-by-R-by-S tensor variates
Z = randn(n,p0);       % n-by-p0 regular design matrix

mu = sum(X .* At, [2,3]) + Z * beta;

err = randn(n,1);
y = mu + err;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. BIC grid search for spline q and n_knot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = 2;                    % a pre_fix cp rank
q_grid = 3:4;             % candidate spline order
n_knot_grid = 3:8;        % candidate number of knots

bic_mat = nan(length(q_grid), length(n_knot_grid));
rss_mat = nan(length(q_grid), length(n_knot_grid));
df_mat  = nan(length(q_grid), length(n_knot_grid));
L_mat   = nan(length(q_grid), length(n_knot_grid));

best_bic = inf;
best_q = nan;
best_n_knot = nan;

best_fit = struct();

for iq = 1:length(q_grid)
    q = q_grid(iq);

    for ik = 1:length(n_knot_grid)
        n_knot = n_knot_grid(ik);

        knots = linspace(0,1,n_knot);
        Bst = bspline_basismatrix(q, [zeros(1,q-1), knots, ones(1,q-1)], t);
        L = size(Bst,2);

        Bs = reshape(Bst,[n,1,1,L]);
        X_Bs = X .* Bs;
        X_Bs = permute(X_Bs, [4,2,3,1]);   % L-R-S-n

        [beta_hat, gamma_hat_cp, glmstats] = kruskal_reg(Z, tensor(X_Bs), y, R, 'normal');
        gamma_hat = double(full(gamma_hat_cp));   % L-R-S
        At_hat = double(ttt(tensor(Bst), tensor(gamma_hat), 2, 1));  % n-R-S

        y_hat = sum(X .* At_hat, [2,3]) + Z * beta_hat;
        err_hat = y - y_hat;

        RSS = sum(err_hat.^2);
        df = p0 + R * (L + R + S);
        BIC = n * log(RSS / n) + df * log(n);

        bic_mat(iq, ik) = BIC;
        rss_mat(iq, ik) = RSS;
        df_mat(iq, ik)  = df;
        L_mat(iq, ik)   = L;

        fprintf('q = %d, n_knot = %d, L = %d, RSS = %.4f, df = %.2f, BIC = %.4f\n', ...
            q, n_knot, L, RSS, df, BIC);

        % update best
        if BIC < best_bic
            best_bic = BIC;
            best_q = q;
            best_n_knot = n_knot;

            best_fit.Bst = Bst;
            best_fit.L = L;
            best_fit.beta_hat = beta_hat;
            best_fit.gamma_hat = gamma_hat;
            best_fit.At_hat = At_hat;
            best_fit.y_hat = y_hat;
            best_fit.err_hat = err_hat;
            best_fit.RSS = RSS;
            best_fit.df = df;
            best_fit.glmstats = glmstats;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Final selected spline model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n========================================\n');
fprintf('Best choice by BIC:\n');
fprintf('q = %d, n_knot = %d, best BIC = %.4f\n', best_q, best_n_knot, best_bic);
fprintf('========================================\n');

q = best_q;
n_knot = best_n_knot;
Bst = best_fit.Bst;
L = best_fit.L;

beta_hat_init = best_fit.beta_hat;
gamma_hat_init = best_fit.gamma_hat;
At_hat_init = best_fit.At_hat;
y_hat_init = best_fit.y_hat;
err_hat_init = best_fit.err_hat;

mae_init = mean(abs(err_hat_init), 'all');
rmse_init = sqrt(mean(err_hat_init.^2, 'all'));
sde_init = std(err_hat_init, 1);

fprintf('Selected spline model:\n');
fprintf('MAE = %.4f\n', mae_init);
fprintf('RMSE = %.4f\n', rmse_init);
fprintf('SDE = %.4f\n', sde_init);


% 4. show BIC table
disp('BIC matrix: rows = q_grid, cols = n_knot_grid');
disp(array2table(bic_mat, ...
    'VariableNames', strcat('knot_', string(n_knot_grid)), ...
    'RowNames', strcat('q_', string(q_grid))));
