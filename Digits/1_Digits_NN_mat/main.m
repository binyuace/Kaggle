
%
%%
% *Part1* :Initialization
clear ; close all; clc


% Read the data

% traindata = csvread('train.csv',1,0);
% test_x = csvread('test.csv',1,0);
% 
% train_x = traindata(1:30000,2:end);
% train_y = traindata(1:30000,1);
% % convert train_y 0 to 10
% train_y(train_y == 0) = 10;
% val_x = traindata(30001:end,2:end);
% val_y = traindata(30001:end,1);
% % convert val_y 0 to 10
% val_y(val_y == 0) = 10;
% sample_submission = csvread('sample_submission.csv',1,0);
% %visualizing the data
% fprintf('Loading and Visualizing Data ...\n')
% 
% m = size(train_y, 1);
% 
% % Randomly select 100 data points to display
% sel = randperm(size(train_x, 1));
% sel = sel(1:100);
% displayData(train_x(sel(1:100), :));
% fprintf('\nProgram paused. Press enter to continue.\n');
% 
% 
% save variable.mat
%%
% part1.5
load variable.mat





%%
% *% Part 2*
%set parameters
fprintf('\nInitializing Neural Network Parameters ...\n')

input_layer_size  = 784;  % 28*28 Input Images of Digits
hidden_layer_size = 50;   % 50 hidden units
num_labels = 10;


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%%
% % part 2.5
% 
% lambda = 0
% [error_train, error_val] = ...
%     bestlambda( lambda,train_x, train_y, val_x, val_y);
% 
% close all;
% plot(hidden_layer_size, error_train,hidden_layer_size, error_val);
% legend('Train', 'Cross Validation');
% xlabel('num_labels');
% ylabel('Error');
% hold
% lambda =3
% [error_train3, error_val3] = ...
%     bestlambda( lambda,train_x, train_y, val_x, val_y);
% plot(hidden_layer_size, error_train3, hidden_layer_size, error_val3);
% legend('Train3', 'Cross Validation3');
% 
% pause;
%% 
% Part 3
% Train Neural network
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
lambda = 3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, train_x, train_y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');


%% ================= Part 4: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');


%% ================= Part 5: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, val_x);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == val_y)) * 100);
pause;
fprintf('pause');
%% ================= Part 6: Implement Predict to test =================
submi = predict(Theta1, Theta2, test_x);
submi(submi ==10) = 0;
submi = [[1:size(submi)]',submi]
submission = array2table(submi,...
    'VariableNames',{'ImageId', 'Label'});

writetable(submission,'submission500iter.csv');



