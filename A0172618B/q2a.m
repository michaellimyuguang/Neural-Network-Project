clear;
clc;
%create 2 directories for storing the plots
mkdir q2a_non_extrapolate;
mkdir q2a_extrapolate;
%parameters
epoch = 500;
no_hidden_neuron = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100];
n = 0.005;
% training data
x_train = [-2:0.05:2]; %given in question
x_train_output = trigo_function(x_train);
% validation data
x_valid = [-2:0.01:2]; %given in question
x_valid_output = trigo_function(x_valid);
% test data
x_test = [-3:0.01:3]; %given in question
x_test_output = trigo_function(x_test);

disp("hold on. plotting in progress.....");
for j = 1:length(no_hidden_neuron)
    text = sprintf("plotting for 1-%d-1.....", no_hidden_neuron(j));
    disp(text);
    %train the neural network.
    net = train_seq(x_train, x_train_output, no_hidden_neuron(j), epoch, n);
    valid_train = net(x_valid);
    test_train = net(x_test);
    name_scatter_valid = sprintf("q2a_non_extrapolate\\1-%d-1", no_hidden_neuron(j));
    name_scatter_test = sprintf("q2a_extrapolate\\1-%d-1", no_hidden_neuron(j));
    %plotting
    %validation data  
    plot(x_valid, x_valid_output);
    hold on;
    scatter(x_valid, valid_train, '.');
    title('Approximation');
    legend({'actual', 'predicted'}, 'Location', 'northeastoutside');
    saveas(gcf, name_scatter_valid, 'png');
    hold off;
    %test data     
    plot(x_test, x_test_output);
    hold on;
    scatter(x_test, test_train, '.');
    title('Extrapolation');
    legend({'actual', 'predicted'}, 'Location', 'northeastoutside');
    saveas(gcf, name_scatter_test, 'png');
    hold off;
end

disp("plotting completed");

function net = train_seq(x_train, x_train_output, no_hidden_neuron, epoch, n)
    x_train_output = num2cell(x_train_output);
    x_train = num2cell(x_train);
%     net = fitnet(no_hidden_neuron, 'traingda'); % training function = traingda
    net = fitnet(no_hidden_neuron, 'traingdx'); % training function = traingdx
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin'; 
    net.trainParam.lr = n;
    for a = 1:epoch
        i = randperm(length(x_train));
        net = adapt(net, x_train(:, i), x_train_output(:, i));      
    end
end
%function given in question
function y = trigo_function(x)
    y = 1.2*sin(pi*x) - cos(2.4*pi*x);
end