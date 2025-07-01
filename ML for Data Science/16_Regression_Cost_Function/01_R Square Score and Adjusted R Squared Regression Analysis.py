'''Generally we have three types of cost function:   (1.) Mean Square Error
                                                     (2.) Mean Absolute Error
                                                     (3.) Root Mean Square Error'''
                                                     
'''[Mean Sqaure Error]:  Mean Square Error(MSE) is the mean squared difference between the actual and predicted values. MSE penalize
                       high errors caused by the outliers by squaring the errors.'''
                       
'''Mean Squared Error = (1/n)[SUM(Yi - Yi')^2]'''      # where y is original data and y' is predicted data 

'''Mean Square Error is also known as L2 Loss and it is a differentiable equation'''

print("----------------------------------------------------------------------------------------------------------------------------------------")


'''[Mean Absolute Error] = It is the mean absolute difference between the actual values and predicted values.
   
   MAE is more robust to outliers. The insensitivity to outliers is because it doesn't penalize high error caused by the outliers'''
   
''' MAE = (1/n)SUM|Yi - Yi'| '''

'''MAE is not an differentiable equation'''

print("----------------------------------------------------------------------------------------------------------------------------------------")

'''[Root Mean Squared Error]: Root Mean Square Error (RMSE) is the root squared mean of the difference between actual and predicted
                              values'''
                              
'''RMSE can be used in situation where we want to penalize high error but not as much as MSE does'''

'''RMSE = Under root of MSE'''




