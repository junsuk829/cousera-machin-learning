function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
thetaSubSum = 0;
for i = 2:size(theta)
    thetaSubSum = thetaSubSum + theta(i)*theta(i);
end


J = (1/m)*( log(sigmoid(theta'*X'))*(-y) -log(1 - sigmoid(theta'*X'))*(1-y) ) + lambda/(2*m)*thetaSubSum;

grad(1)=(1/m)*((sigmoid(theta'*X') - y')*X(:,1));
for i = 2:size(grad)
    grad(i)=(1/m)*((sigmoid(theta'*X') - y')*X(:,i)) + lambda/m*theta(i);
end


% =============================================================

end
