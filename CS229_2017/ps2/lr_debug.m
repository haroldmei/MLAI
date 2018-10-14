% compute gradient given X, y, and theta
compute_grad = @(X, y, theta) (-1 / size(X, 1)) * X' * (y ./(1 + exp((X * theta).* y)));

%% training for dataset A
disp('============= Training model on dataset A ============');
data = load('data_a.txt');
n = size(data, 1);
% adding intercept to data X
X = [ones(n, 1), data(:, 2:end)];
y = data(:, 1);

% training a logistic model
theta = zeros(size(X, 2), 1);
learning_rate = 10;
for i = 1:10^9
    prev_theta = theta;
    grad = compute_grad(X, y, theta);
    theta = theta - learning_rate * grad;
    if mod(i, 10000) == 0
        fprintf('finished iteration %d \n', i);    
    end
    if norm(theta - prev_theta) < 10^-15
        fprintf('converged in %d iterations \n', i-1);
        break;
    end
end

%% training for dataset B
disp('============= Training model on dataset A ============');
data = load('data_b.txt');
n = size(data, 1);
% adding intercept to data X
X = [ones(n, 1), data(:, 2:end)];
y = data(:, 1);

% training a logistic model
theta = zeros(size(X, 2), 1);
learning_rate = 10;
for i = 1:10^9
    prev_theta = theta;
    grad = compute_grad(X, y, theta);
    theta = theta - learning_rate * grad;
    if mod(i, 10000) == 0
        fprintf('finished iteration %d \n', i);    
    end
    if norm(theta - prev_theta) < 10^-15
        fprintf('converged in %d iterations \n', i-1);
        break;
    end
end
