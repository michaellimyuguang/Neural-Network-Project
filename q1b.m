clear;
clc;

%define parameters
W = [0.1; 0.1]; %starting point for x and y
x = zeros(); %used for plotting
y = zeros(); %used for plotting
iteration_list = zeros(); %used for plotting
error_list = zeros();
stop = 0;
flag = 1;
max_iters = 1e20; %maximum iterations
j = 1;

while (flag && (j <= max_iters))  
    x(j) = W(1);
    y(j) = W(2);
    W_prev = W;
    W_delta = -inv(hessian_matrix(W(1), W(2))) * rosenbrock_grad(W(1), W(2)); %formula given in the question
    W = W + W_delta;
    error = norm(W - W_prev);
    error_list(j) = error;
    iteration_list(j) = j; %used for plotting
    if (stop >= error)
        flag = 0; %flag toggle to terminate the loop
    end
    j = j + 1;
end

%plotting
figure(1);
x_range = [-2:0.01:2]; %x range of -2 to 2 in steps 0.01
y_range = [-2:0.01:2]; %y range of -2 to 2 in steps 0.01
f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2; %given formula for f(x,y) in question
[X, Y] = meshgrid(x_range, y_range);
FF = f(X,Y);
f_xy = f(x, y);
contour(X, Y, FF, 100);
hold on;
plot(x, y, 'm');
xlabel('x');
ylabel('y');
title('trajectory of (x,y) in the 2D space with contour');
hold off;

figure(2);
plot(iteration_list, x, iteration_list, y);
xlabel('iterations');
ylabel('x/y');
title('plot of x/y vs iterations');
legend({'x', 'y'}, 'Location', 'northeastoutside');

figure(3);
plot(iteration_list, f_xy);
xlabel('iterations');
ylabel('f(x,y)');
title('plot of f(x,y) vs iterations');

figure(4);
plot(x, y);
xlabel('x');
ylabel('y');
title('trajectory of (x,y) in the 2D space without contour');

%function from 1(a)
function grad = rosenbrock_grad(x, y)
    grad = [(-2*(1-x))-(400*x*(y-(x^2))); 200*(y-(x^2))];
end
%function to calculate the hessian matrix
function H = hessian_matrix(x, y)
    df2dx2 = (1200*(x^2)) - (400*y) + 2;
    df2dy2 = 200;
    df2dxdy = -400 * x;
    df2dydx = -400 * x;
    H = [df2dx2, df2dxdy; df2dydx, df2dy2];
end