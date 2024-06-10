function R2 = rsquare(Y_pred,Y)
if length(Y_pred) ~= length(Y), error('Vector should be of same length');end
% Calculation of R2 according to the formula: SSreg/SStot
SSreg = sum((Y_pred - mean(Y)).^2);
SStot = sum((Y - mean(Y)).^2);
R2 =SSreg/SStot;
% Output limitations
if R2 > 1, error('Irregular value, check your data');end
if R2 < 0, error('Irregular value, check your data');end
end