#!/bin/sh 
cd Test#1_x-65_CrossValidation3Batches_WithoutBalance && python exp_executor.py
cd ../
cd Test#2_x-65_CrossValidation3Batches(Danil)_BalancedData && python exp_executor.py
cd ../
cd Test#3_x-65_CrossValidation3Batches(Artem)_BalancedData && python exp_executor.py
