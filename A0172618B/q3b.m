clear;
clc;
%create directory to store the plots for sequential mode
mkdir q3b_sequential_mode;
%parameters
dim = [16, 64, 101];
epochs = 2000;
n_train = 1000;
n_val = 250;

disp("hold on. plotting in progress.....");
for k = 1:length(dim)
    %training data store in matrix
    train_filepath = "TrainImages";
    train_image = dir(fullfile(train_filepath, "*.jpg")); %folder for images
    train_att = dir(fullfile(train_filepath, "*.att")); %folder for attributes
    training_data = zeros([(dim(k)*dim(k)), n_train]); 
    training_label = zeros([1, n_train]);
    for i = 1:n_train
        img = extract_image(train_filepath, train_image, i, dim(k));
        label = extract_label(train_filepath, train_att, i);
        training_data(:, i) = img;
        training_label(:, i) = label;
    end
    %validation data store in matrix
    val_filepath = "TestImages";
    val_image = dir(fullfile(val_filepath, "*.jpg")); %folder for images
    val_att = dir(fullfile(val_filepath, "*.att")); %folder for attributes
    validation_data = zeros([(dim(k)*dim(k)), n_val]);
    validation_label = zeros([1, n_val]);
    for i = 1:n_val
        img = extract_image(val_filepath, val_image, i, dim(k));
        label = extract_label(val_filepath, val_att, i);
        validation_data(:,i) = img;
        validation_label(:,i) = label;
    end
    
    n_component = extract_component(train_filepath, train_image, n_train, dim(k));
    training_data_pca = zeros([dim(k)^2, n_train]);
    for w = 1:n_train
        X = img_pca(train_filepath, train_image, n_component, dim(k), w);
        training_data_pca(:,w) = X(:);
    end
    
    [net, accu_train_pca, accu_val_pca] = train_seq(training_data_pca, training_label, validation_data, validation_label, epochs);
    epoch_plot = [1:1:epochs];
    fprintf("plotting for %dx%d and %d components\n", dim(k), dim(k), n_component);
    name = sprintf("q3b_sequential_mode\\%dx%d_%d_components", dim(k), dim(k), n_component);
    %plotting
    figure;
    plot(epoch_plot, accu_train_pca, epoch_plot, accu_val_pca);
    xlabel("epoch");
    ylabel("accuracy (%)");
    legend({'training', 'validation'}, 'Location', 'northeastoutside');
    saveas(gcf, name, 'png');
end
disp("plotting completed");

function [net, accu_train, accu_val] = train_seq(training_data, training_label, validation_data, validation_label, epochs)
    net = perceptron;
    net.trainParam.epochs = epochs;
    accu_train = zeros(1, epochs);
    accu_val = zeros(1, epochs);
    for k = 1:epochs
        index = randperm(size(training_data, 2));
        net = adapt(net, training_data(:,index), training_label(:,index));
        pred_train = net(training_data(:,index));
        accu_train(k) = (1 - mean(abs(pred_train - training_label(:,index)))) * 100;
        val_train = net(validation_data);
        accu_val(k) = (1 - mean(abs(val_train - validation_label))) * 100;
    end
end

%function to extract images
function img = extract_image(filepath, files, i, dim)
    filename = filepath + '\\' + files(i).name;
    I = imread(filename);
    G = rgb2gray(I);
    G = imresize(G, [dim dim]);
    img = G(:); %V = G(:);
end

%function to extract labels
function label = extract_label(filepath, files, i)
    filename = filepath + '\\' + files(i).name;
    L = load(filename);
    label = L(1); %1 represents my group ID. 
end

function X = img_pca(filepath, files, component, dim, w)
    filename = filepath + '\\' + files(w).name;
    I = imread(filename);
    img_old = rgb2gray(I);
    img = double(img_old);
    img = imresize(img, [dim dim]);
    img_mean = mean(img);
    img_adjusted = img - img_mean;    
    [coeff, score] = pca(img_adjusted);
    X = score(:,1:component-1)*coeff(:,1:component-1)';
    X = X + img_mean;
    X = uint8(X);   
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