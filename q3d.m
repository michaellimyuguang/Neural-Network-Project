clear;
clc;
%create directory to store the plots for sequential mode
mkdir q3d_sequential_mode;

%parameters
n_train = 1000;
n_val = 250;
epochs = 500;

%training data store in matrix
train_filepath = "TrainImages";
train_image = dir(fullfile(train_filepath, "*.jpg")); %folder for images
train_att = dir(fullfile(train_filepath, "*.att")); %folder for attributes
training_data = zeros([10201, n_train]); %101*101
training_label = zeros([1, n_train]);

disp("hold on. plotting in progress.....");

for i = 1:n_train
    img = extract_image(train_filepath, train_image, i);
    label = extract_label(train_filepath, train_att, i);
    training_data(:, i) = img;
    training_label(:, i) = label;
end

%validation data store in matrix
val_filepath = "TestImages";
val_image = dir(fullfile(val_filepath, "*.jpg")); %folder for images
val_att = dir(fullfile(val_filepath, "*.att")); %folder for attributes
validation_data = zeros([10201, n_val]); %101*101
validation_label = zeros([1, n_val]);
for i = 1:n_val
    img = extract_image(val_filepath, val_image, i);
    label = extract_label(val_filepath, val_att, i);
    validation_data(:,i) = img;
    validation_label(:,i) = label;
end

n_component = extract_component(train_filepath, train_image, n_train, 101);
components_list = [ceil(n_component/20), ceil(n_component/2), n_component];
for k = 1:length(components_list)
   fprintf("components for testing: %d\n", components_list(k));
   filename = sprintf("q3d_sequential_mode\\%d_components", components_list(k));
   [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_label, epochs, filename, components_list(k));
end

disp("plotting completed");

function [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_label, epochs, filename, component)
    % define patternnet
    net = patternnet(component);
    net.trainFcn = 'trainscg';
    net.trainParam.lr = 0.01;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'logsig';
    net.trainParam.epochs=epochs;
    epoch_plot = [1:1:epochs];
    accu_train = zeros(1, epochs);
    accu_val = zeros(1, epochs);
    % Training loop
    for i = 1:epochs
        index = randperm(size(training_data, 2));
        net = adapt(net, training_data(:,index), training_label(:,index));

        pred_train = net(training_data(:,index));
        accu_train(i) = (1 - mean(abs(pred_train - training_label(:,index)))) * 100;
        
        val_train = net(validation_data);
        accu_val(i) = (1 - mean(abs(val_train - validation_label))) * 100;
    end
    figure;
    plot(epoch_plot, accu_train, epoch_plot, accu_val);
    xlabel("epochs");
    ylabel("accuracy (%)");
    legend({'training', 'validation'}, 'Location', 'southeast');
    saveas(gcf, filename, 'png');
end

%function to extract images
function img = extract_image(filepath, files, i)
    filename = filepath + '\\' + files(i).name;
    I = imread(filename);
    G = rgb2gray(I);
    if size(G) ~= [101 101]
        old_dim = size(G);
        G = imresize(G, [101 101]);
        new_dim = size(G);
        %show the image files that has been resized to 101x101
        fprintf("filename: %s\n", filename);
        fprintf("old dimension: %s\n", mat2str(old_dim));
        fprintf("new dimension: %s\n", mat2str(new_dim));
    end
    img = G(:); %V = G(:);
end
%function to extract labels
function label = extract_label(filepath, files, i)
    filename = filepath + '\\' + files(i).name;
    L = load(filename);
    label = L(1); %1 represents my group ID. 
end

function n_component = extract_component(filepath, files, n_total, dim)
    eff_rank = zeros([1 n_total]);
    for i = 1:n_total
        Irgb = imread(filepath + '\\' + files(i).name);
        I = rgb2gray(Irgb);
        singular_val = svd(double(I));
        sv_sum = 0;
        ksv_sum = 0;
        for j = 1:dim
            sv_sum = sv_sum + singular_val(j);
        end
        for k = 1:dim
            ksv_sum = ksv_sum + singular_val(k);
            if (ksv_sum/sv_sum) >= 0.99
                eff_rank(i) = k;
                break
            end
        end
    end
    n_component = ceil(mean(eff_rank));
end