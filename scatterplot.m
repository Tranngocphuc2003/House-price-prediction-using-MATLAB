function scatter_plot = scatterplot(X,Y,Y_predict,varname)
last_atribute = length(varname)-1;
for i= 1:last_atribute
    subplot(3,2,i);
    scatter(X(:,i),Y);
    hold on
    scatter(X(:,i),Y_predict);
    xlabel(varname(i));
    ylabel(varname(end));
    hold off
 end
end