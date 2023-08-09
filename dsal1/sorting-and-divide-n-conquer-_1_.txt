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

# In[58]:


# List of numbers in random order
test0{
    'input': {
        'nums':[6,7,5,3,5,1,2,4,7,0,7,4]
    },
    'output': [0, 1, 2, 3, 4, 5,5,]
}


# In[59]:


# List of numbers in random order
test1 = {
    'input':{
        'nums':[78, 9, 4, 5, 0, -45, -95, 1, 2, 6, 63, -98, -768, -187]
    },
    'output': [-768, -187, -98, -95,-45, 0, 1, 2, 4, 5, 6, 9, 63, 78]
}


# In[60]:


# A list that is already sorted
test2 = {
    'input':{
        'nums': [1, 2, 3, 5, 7, 9, 11, 17, 19]
    },
    'output': [1, 2, 3, 5, 7, 9, 11, 17, 19]
}


# In[61]:


# A list that is sorted in descending order.
test3 = {
    'input':{
        'nums': [99, 97, 96, 87, 85, 78, 55, 10, 7, 5, 4, 3, 2, 1, 0]
    },
    'output': [0, 1, 2, 3, 4, 5, 7, 10, 55, 78, 85, 87, 96, 97, 99]
}


# In[62]:


# An empty list
test4 = {
    'input':{
        'nums': []
    },
    'output': []
}


# In[63]:


# A list containing just one element 
test5 = {
    'input':{
        'nums': [90]
    },
    'output': [90]
}


# In[64]:


# A list containing one element repeated many times
test6 = {
    'input':{
        'nums': [78, 78, 78, 78, 78]
    },
    'output': [78, 78, 78, 78, 78]
}


# In[65]:


# A list containig repeating elements 
test7 = {
    'input':{
        'nums': [5, -12, 2, 6, 1, 23, 7, 7, -12, 6, 12, 1, -243, 1, 0]
    },
    'output': [-243, -12, -12, 0, 1, 1, 1, 2, 5, 6, 6, 7, 7, 12, 23]
}


# To create the final test case (a really long list), we can start with a sorted list created using `range` and shuffle it to create the input.
# 

# In[66]:


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


# In[67]:


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

# In[68]:


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

# In[69]:


x, y = 2, 3
x, y = y, x
x, y


# Let's test it with an example.

# In[70]:


nums0, output0 = test0['input']['nums'], test0['output']

print('Input:', nums0)
print('Expected output:', output0)
result0 = bubble_sort(nums0)
print('Actual output:', result0)
print('Match:', result0 == output0)


# In[71]:


result0 ==  output0


# We can evaluate all the cases together using the `evaluate_test_cases` helper function from the `jovian` library.

# In[72]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[73]:


from jovian.pythondsa import evaluate_test_cases


# In[74]:


from jovian.pythondsa import evaluate_test_cases


# In[75]:


results = evaluate_test_cases(bubble_sort, tests)


# In[76]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[77]:


import jovian


# In[78]:


jovian.commit()


# #
# Analyze the algorithm's complexity and identify inefficiencies
# The core operations in bubble sort are "compare" and "swap". To analyze the time complexity, we can simply count the total number of comparisons being made, since the total number of swaps will be less than or equal to the total number of comparisons (can you see why?).
# 
# for _ in range(len(nums) - 1):
#     for i in range(len(nums) - 1):
#         if nums[i] > nums[i+1]:
#             nums[i], nums[i+1] = nums[i+1], nums[i]
# There are two loops, each of length n-1, where n is the number of elements in nums. So the total number of comparisons is  (𝑛−1)∗(𝑛−1)
#   i.e.  (𝑛−1)2
#   i.e.  𝑛2−2𝑛+1
#  .
# 
# Expressing this in the Big O notation, we can conclude that the time complexity of bubble sort is  𝑂(𝑛2)
#   (also known as quadratic complexity).
# 
# Exercise: Verify that the bubble sort requires  𝑂(1)
#   additional space.
# 
# The space complexity of bubble sort is  𝑂(𝑛)
#  , even thought it requires only constant/zero additional space, because the space required to store the inputs is also considered while calculating space complexity.
# 
# As we saw from the last test, a list of 10,000 numbers takes about 12 seconds to be sorted using bubble sort. A list of ten times the size will 100 times longer i.e. about 20 minutes to be sorted, which is quite inefficient. A list of a million elements would take close to 2 days to be sorted.
# 
# 

# ### Insertion Sort
# 
# Before we look at explore more efficient sorting techniques, here's another simple sorting technique called insertion sort, where we keep the initial portion of the array sorted and insert the remaining elements one by one at the right position.

# In[86]:


def insertion_sort(nums):
    nums = list(nums)
    for i in range(len(nums)):
        cur = nums.pop(i)
        j = i-1
        while j >=0 and nums[j] > cur:
            j -= 1
        nums.insert(j+1, cur)
    return nums            


# In[87]:


nums0, output0 = test0['input']['nums'], test0['output']

print('Input:', nums0)
print('Expected output:', output0)
result0 = insertion_sort(nums0)
print('Actual output:', result0)
print('Match:', result0 == output0)


# 
# 
# 
# To performing sorting more efficiently, we'll apply a strategy called **Divide and Conquer**, which has the following general steps:
# 
# 1. Divide the inputs into two roughly equal parts.
# 2. Recursively solve the problem individually for each of the two parts.
# 3. Combine the results to solve the problem for the original inputs.
# 4. Include terminating conditions for small or indivisible inputs.
# 
# Here's a visual representation of the strategy:
# 
# ![](https://www.educative.io/api/edpresso/shot/5327356208087040/image/6475288173084672)
# 
# ### Merge Sort
# 
# Following a visual representation of the divide and conquer applied for sorting numbers. This algorithm is known as merge sort:
# 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Merge_sort_algorithm_diagram.svg/2560px-Merge_sort_algorithm_diagram.svg.png" width="480">
# 
# 

# 
# 
# Here's a step-by-step description for merge sort:
# 
# 1. If the input list is empty or contains just one element, it is already sorted. Return it.
# 2. If not, divide the list of numbers into two roughly equal parts.
# 3. Sort each part recursively using the merge sort algorithm. You'll get back two sorted lists.
# 4. Merge the two sorted lists to get a single sorted list
# 
# 
# 
# 
# 
# 
# T

# In[88]:


def merge_sort(nums):
    # Terminating condition (list of 0 or 1 elements)
    if len(nums) <= 1:
        return nums
    
    # Get the midpoint
    mid = len(nums) // 2
    
    # Split the list into two halves
    left = nums[:mid]
    right = nums[mid:]
    
    # Solve the problem for each half recursively
    left_sorted, right_sorted = merge_sort(left), merge_sort(right)
    
    # Combine the results of the two halves
    sorted_nums =  merge(left_sorted, right_sorted)
    
    return sorted_nums


# In[89]:


def merge(nums1, nums2):    
    # List to store the results 
    merged = []
    
    # Indices for iteration
    i, j = 0, 0
    
    # Loop over the two lists
    while i < len(nums1) and j < len(nums2):        
        
        # Include the smaller element in the result and move to next element
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1 
        else:
            merged.append(nums2[j])
            j += 1
    
    # Get the remaining parts
    nums1_tail = nums1[i:]
    nums2_tail = nums2[j:]
    
    # Return the final merged array
    return merged + nums1_tail + nums2_tail


# Let's test the merge operation, before we test merge sort. 

# In[83]:


merge([1, 4, 7, 9, 11], [-1, 0, 2, 3, 8, 12])


# 
# It seems to work as expected. We can now test the merge_sort function.

# In[84]:


nums0, output0 = test0['input']['nums'], test0['output']

print('Input:', nums0)
print('Expected output:', output0)
result0 = merge_sort(nums0)
print('Actual output:', result0)
print('Match:', result0 == output0)


# In[85]:


results = evaluate_test_cases(merge_sort, tests)


# In[90]:


jovian.commit()


# # quick sort

# To overcome the space inefficiencies of merge sort, we'll study another divide-and-conquer based sorting algorithm called **quicksort**, which works as follows:
# 
# 1. If the list is empty or has just one element, return it. It's already sorted.
# 2. Pick a random element from the list. This element is called a _pivot_.
# 3. Reorder the list so that all elements with values less than or equal to the pivot come before the pivot, while all elements with values greater than the pivot come after it. This operation is called _partitioning_.
# 4. The pivot element divides the array into two parts which can be sorted independently by making a recursive call to quicksort.

# In[94]:


def quicksort(nums, start=0, end=None):
    # print('quicksort', nums, start, end)
    if end is None:
        nums = list(nums)
        end = len(nums) - 1
    
    if start < end:
        pivot = partition(nums, start, end)
        quicksort(nums, start, pivot-1)
        quicksort(nums, pivot+1, end)

    return nums


# Here's an implementation of partition, which uses the last element of the list as a pivot:

# In[96]:


def partition(nums, start=0, end=None):
    # print('partition', nums, start, end)
    if end is None:
        end = len(nums) - 1
    
    # Initialize right and left pointers
    l, r = start, end-1
    
    # Iterate while they're apart
    while r > l:
        # print('  ', nums, l, r)
        # Increment left pointer if number is less or equal to pivot
        if nums[l] <= nums[end]:
            l += 1
        
        # Decrement right pointer if number is greater than pivot
        elif nums[r] > nums[end]:
            r -= 1
        
        # Two out-of-place elements found, swap them
        else:
            nums[l], nums[r] = nums[r], nums[l]
    # print('  ', nums, l, r)
    # Place the pivot between the two parts
    if nums[l] > nums[end]:
        nums[l], nums[end] = nums[end], nums[l]
        return l
    else:
        return end


# In[97]:


l1 = [1, 5, 6, 2, 0, 11, 3]
pivot = partition(l1)
print(l1, pivot)


# In[98]:


nums0, output0 = test0['input']['nums'], test0['output']

print('Input:', nums0)
print('Expected output:', output0)
result0 = quicksort(nums0)
print('Actual output:', result0)
print('Match:', result0 == output0)


# test all the cases using the evaluate_test_cases function from jovian.

# In[100]:


from jovian.pythondsa import evaluate_test_cases

results = evaluate_test_cases(quicksort, tests)


# In[ ]:





# In[ ]:




