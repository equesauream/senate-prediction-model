%% model for predicting senators voting patterns

function z = prox_l1(x, t)
% proximal gradient for t * ||x||_1
    [n, ~] = size(x);
    z = zeros(n, 1);
    z(x < -t) = x(x < -t) + t;
    z(x > t) = x(x > t) - t;
end

function z = prox_linf(x, t)
% proximal gradient for t * ||x||_inf
    [n, ~] = size(x);

    [~, p] = sort(abs(x), 'desc');
    x_sorted = x(p);
    c = 0;
    z = zeros(n, 1);
    
    for i = 1:n
        c = c + abs(x_sorted(i));
        if i == n
            y = 0;
        else
            y = abs(x_sorted(i + 1));
        end
        if c >= t + i * y
            z(1:i) = sgn(x_sorted(1:i)) * (c - t)/i;
            z((i + 1):n) = x_sorted((i + 1) : n);
            break;
        end
    end

    [~, q] = sort(p);
    z = z(q);

    function u = sgn(v)
        u = v;
        u(v < 0) = -1;
        u(v >= 0) = 1;
    end
end

function z = prox_l2sq(x, t)
% proximal gradient for t * ||x||^2
    z = (1/(1 + 2 * t)) * x;
end

%% Beginning of driver script

A = prepare_senatedata();

% M = # of bills, N = # of senators
[M, N] = size(A);

% train on the first 6 senators, test on the remainder
num_training = 6;

%% APGD parameters
gamma = 1;
tol = 1e-6;
maxit = 1e7;

%% training & testing procedure
fprintf('using gamma = %i, tol = %2s, maxit = %2s\n\n', gamma, tol, maxit);
fprintf('%12s','pass','l1','linf', 'l2sq', 'count');
fprintf('\n')
for pass = 2:N
    count = height(ytraining);
    Atraining = A(1:num_training, 1:(pass - 1));
    ytraining = A(1:num_training, pass);

    L = max(eig(Atraining' * Atraining));
    x0 = zeros(pass - 1, 1);

    [x_l1, ~] = apgd(@(x) Atraining' * (Atraining * x - ytraining), @prox_l1, gamma, L, x0, tol, maxit);
    [x_linf, ~] = apgd(@(x) Atraining' * (Atraining * x - ytraining), @prox_linf, gamma, L, x0, tol, maxit);
    [x_l2sq, ~] = apgd(@(x) Atraining' * (Atraining * x - ytraining), @prox_l2sq, gamma, L, x0, tol, maxit);

    Atest = A((num_training + 1):end, 1:(pass - 1));
    ytraining = A((num_training + 1):end, pass);

    b_l1 = sign(Atest * x_l1) - ytraining;
    b_linf = sign(Atest * x_linf) - ytraining;
    b_l2sq = sign(Atest * x_l2sq) - ytraining;

    fprintf('%12i', pass)
	fprintf('%12i', height(b_l1(b_l1 == 0)), height(b_linf(b_linf == 0)), height(b_l2sq(b_l2sq == 0)), count)
    fprintf('\n')

    scores(1, pass - 1) = height(b_l1(b_l1 == 0));
    scores(2, pass - 1) = height(b_linf(b_linf == 0));
    scores(3, pass - 1) = height(b_l2sq(b_l2sq == 0));
end

tcount = count * (N - 1);
fprintf('%12s', 'total')
fprintf('%12i', sum(scores(1, :)), sum(scores(2, :)), sum(scores(3, :)), tcount)
fprintf('\n')

fprintf('%12s', 'percent')
fprintf('%12f', sum(scores(1, :))/tcount, sum(scores(2, :))/tcount, sum(scores(3, :))/tcount)
fprintf('\n')