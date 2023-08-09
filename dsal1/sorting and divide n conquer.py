#!/usr/bin/env python
# coding: utf-8

# # sorting Algorithm

#  ## Problem 
# 
# 
# In this notebook, we'll focus on solving the following problem:
# 
# > **QUESTION 1**: You're working on a new feature on Jovian called "Top Notebooks of the Week". Write a function to sort a list of notebooks in decreasing order of likes. Keep in mind that up to millions of notebooks  can be created every week, so your function needs to be as efficient as possible.
# 
# 
# The problem of sorting a list of objects comes up over and over in computer science and software development, and it's important to understand common approaches for sorting, and the trade-offs they offer. Before we solve the above problem, we'll solve a simplified version of the problem:
# 
# > **QUESTION 2**: Write a program to sort a list of numbers.
# 
# 
# "Sorting" usually refers to "sorting in ascending order", unless specified otherwise.

# ## 1. State the problem clearly. Identify the input & output formats.
# 
# 
# #### Problem
# 
# > We need to write a function to sort a list of numbers in increasing order.
# 
# #### Input
# 
# 1. `nums`: A list of numbers e.g. `[4, 2, 6, 3, 4, 6, 2, 1]` 
# 
# #### Output
# 
# 2. `sorted_nums`: The sorted version of `nums` e.g. `[1, 2, 2, 3, 4, 4, 6, 6]`
# 
# 
# The signature of our function would be as follows:

# In[32]:


def sort(nums):
    pass


# ## 2. Take  some example inputs & outputs. 
# 
# Here are some scenarios we may want to test out:
# 
# 1. Some lists of numbers in random order.
# 2. A list that's already sorted.
# 3. A list that's sorted in descending order.
# 4. A list containing repeating elements.
# 5. An empty list. 
# 6. A list containing just one element.
# 7. A list containing one element repeated many times.
# 8. A really long list.
# 
# Let's create some test cases for these scenarios. We'll represent each test case as a dictionary for easier automated testing.

# In[33]:


# List of numbers in random order
test0 = {
    'input': {
        'nums':[6,7,5,3,5,1,2,4,7,0,7,4]
    },
    'output': [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7]
}


# In[34]:


# List of numbers in random order
test1 = {
    'input':{
        'nums':[78, 9, 4, 5, 0, -45, -95, 1, 2, 6, 63, -98, -768, -187]
    },
    'output': [-768, -187, -98, -95, 0, 1, 2, 4, 5, 6, 9, 63, 78]
}


# In[35]:


# A list that is already sorted
test2 = {
    'input':{
        'nums': [1, 2, 3, 5, 7, 9, 11, 17, 19]
    },
    'output': [1, 2, 3, 5, 7, 9, 11, 17, 19]
}


# In[36]:


# A list that is sorted in descending order.
test3 = {
    'input':{
        'nums': [99, 97, 96, 87, 85, 78, 55, 10, 7, 5, 4, 3, 2, 1, 0]
    },
    'output': [0, 1, 2, 3, 4, 5, 7, 10, 55, 78, 85, 87, 96, 97, 99]
}


# In[37]:


# An empty list
test4 = {
    'input':{
        'nums': []
    },
    'output': []
}


# In[38]:


# A list containing just one element 
test5 = {
    'input':{
        'nums': [90]
    },
    'output': [90]
}


# In[39]:


# A list containing one element repeated many times
test6 = {
    'input':{
        'nums': [78, 78, 78, 78, 78]
    },
    'output': [78, 78, 78, 78, 78]
}


# In[40]:


# A list containig repeating elements 
test7 = {
    'input':{
        'nums': [5, -12, 2, 6, 1, 23, 7, 7, -12, 6, 12, 1, -243, 1, 0]
    },
    'output': [-243, -12, -12, 0, 1, 1, 1, 2, 5, 6, 6, 7, 7, 12, 23]
}


# To create the final test case (a really long list), we can start with a sorted list created using `range` and shuffle it to create the input.
# 

# In[41]:


import random


in_list = list(range(10000))
out_list =list(range(10000))
random.shuffle(in_list)

test8 = {
    'input':{
        'nums': in_list
    },
    'output': out_list
}


# In[42]:


tests= [test0,test1, test2, test3, test4, test5, test6, test7]


# # Implement the solution and test it using example inputs.
# 
# The implementation is straight forward. We'll create a copy of the list inside our function, to avoid changing it while sorting.
# 
# 
# It's easy to come up with a correct solution. Here's one: 
# 
# 1. Iterate over the list of numbers, starting from the left
# 2. Compare each number with the number that follows it
# 3. If the number is greater than the one that follows it, swap the two elements
# 4. Repeat steps 1 to 3 till the list is sorted.

# In[43]:


def bubble_sort(nums):
    # Create a copy of the list, to avoid changing it
    nums = list(nums)
    
    # 4. Repeat the process n-1 times
    for _ in range(len(nums) - 1):
        
        # 1. Iterate over the array (except last element)
        for i in range(len(nums) - 1):
            
            # 2. Compare the number with  
            if nums[i] > nums[i+1]:
                
                # 3. Swap the two elements
                nums[i], nums[i+1] = nums[i+1], nums[i]
    
    # Return the sorted list
    return nums


# Notice how we're a tuple assignment to swap two elements in a single line of code

# In[44]:


x, y = 2, 3
x, y = y, x
x, y


# Let's test it with an example.

# In[45]:


nums0, output0 = test0['input']['nums'], test0['output']

print('Input:', nums0)
print('Expected output:', output0)
result0 = bubble_sort(nums0)
print('Actual output:', result0)
print('Match:', result0 == output0)


# In[46]:


result0 ==  output0


# We can evaluate all the cases together using the `evaluate_test_cases` helper function from the `jovian` library.

# In[47]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[48]:


from jovian.pythondsa import evaluate_test_cases


# In[49]:


from jovian.pythondsa import evaluate_test_cases


# In[50]:


results = evaluate_test_cases(bubble_sort, tests)


# In[ ]:




