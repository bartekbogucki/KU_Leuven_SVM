X1 = randn(50, 2) + 1;
X2 = randn(51, 2) - 1;

Y1 = ones(50, 1);
Y2 = -ones(51, 1);

figure;
hold on;

% Plot the two classes
plot(X1(:,1), X1(:,2), 'ro', 'DisplayName', 'Class 1 (+1)');
plot(X2(:,1), X2(:,2), 'bo', 'DisplayName', 'Class 2 (-1)');

% Plot the decision boundary: x1 + x2 = 0 (i.e., x2 = -x1)
xline_vals = -4:0.1:4;
plot(xline_vals, -xline_vals, 'k-', 'LineWidth', 1.5, 'DisplayName', 'x₁ + x₂ = 0');

% Add axis labels and formatting
xlabel('x₁');
ylabel('x₂');
title(['Two Gaussian classes with an optimal classification line']);
legend('Location', 'best');
axis equal;
grid on;

% Bound the plot
xlim([-4, 4]);
ylim([-4, 4]);

hold off;
