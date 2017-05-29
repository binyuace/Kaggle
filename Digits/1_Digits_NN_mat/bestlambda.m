function [error_train, error_val] = ...
    bestlambda(lambda,train_x, train_y, val_x, val_y)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%
input_layer_size  = 784;  % 28*28 Input Images of Digits
% hidden_layer_size = 50;   % 50 hidden units
num_labels = 10;

% Selected values of lambda (you should not change this)
% You need to return these variables correctly.
error_train = zeros(7, 1);
error_val = zeros(7, 1);
i=0
for hidden_layer_size=[20,30,35,40,45,50,60]
    i=i+1;
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    options = optimset('MaxIter', 100);

    %  You should also try different values of lambda

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
    
    
    pred1 = mean(double(predict(Theta1, Theta2, train_x) == train_y)) * 100
    error_train(i,1) = pred1;
   
    pred2 = mean(double(predict(Theta1, Theta2, val_x) == val_y)) * 100
    error_val(i,1) = pred2;
   
end


end

