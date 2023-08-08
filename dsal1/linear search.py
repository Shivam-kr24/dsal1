#!/usr/bin/env python
# coding: utf-8

# In[1]:


def locate_card(cards, query): # create the signature of our function
    pass # in python we cannotput empty body .


# test cases

# In[2]:


cards = [13,11,10,7,4,3,1,0]
query = 7
output = 3


# we can test our function by passing the inputs into functions and compare with the expected output 

# In[3]:


result = locate_card(cards, query)
print(result)


# In[4]:


result == output


# we'll represent our test cases as dictionaries to make it easier .
# for example the above test cases can be represented

# In[5]:


test = {
    'input': {
        'cards':[13, 11, 10, 7, 4, 3, 2, 1, 0 ],
    'query':7
    },
    'output' : 3
    
}


# the function can now be tested as follows

# In[6]:


locate_card(**test['input']) == test['output']


# Our function should be able to handle any set of valid inputs we pass into it. Here's a list of some possible variations we might encounter:
# 
# 1. The number `query` occurs somewhere in the middle of
#  the list  `cards`.
# 2. `query` is the first element in `cards`.
# 3. `query` is the last element in `cards`.
# 4. The list `cards` contains just one element, which is `query`.
# 5. The list `cards` does not contain number `query`.
# 6. The list `cards` is empty.
# 7. The list `cards` contains repeating numbers.
# 8. The number `query` occurs at more than one position in `cards`.
# 9. (can you think of any more variations?)
# 
# > **Edge Cases**: It's likely that you didn't think of all of the above cases when you read the problem for the first time. Some of these (like the empty array or `query` not occurring in `cards`) are called *edge cases*, as they represent rare or extreme examples. 
# 
# While edge cases may not occur frequently, your programs should be able to handle all edge cases, otherwise they may fail in unexpected ways. Let's create some more test cases for the variations listed above. We'll store all our test cases in an list for easier testing.

# In[7]:


tests = []


# In[8]:


#query occurs in the middle 
tests.append(test)

tests.append({
    'input': {
        'cards': [13, 11, 10, 7, 4, 3, 1, 0],
        'query': 1
    },
    'output': 6
})


# In[9]:


# query is the first element
tests.append({
    'input': {
        'cards': [4, 2, 1, -1],
        'query': 4
    },
    'output': 0
})


# In[10]:


# query is the last element
tests.append({
    'input': {
        'cards': [3, -1, -9, -127],
        'query': -127
    },
    'output': 3
})


# In[11]:


# cards contains just one element, query
tests.append({
    'input': {
        'cards': [6],
        'query': 6
    },
    'output': 0 
})


# The problem statement does not specify what to do if the list `cards` does not contain the number `query`. 
# 
# 1. Read the problem statement again, carefully.
# 2. Look through the examples provided with the problem.
# 3. Ask the interviewer/platform for a clarification.
# 4. Make a reasonable assumption, state it and move forward.
# 
# We will assume that our function will return `-1` in case `cards` does not contain `query`.

# In[12]:


# cards does not contain query 
tests.append({
    'input': {
        'cards': [9, 7, 5, 2, -9],
        'query': 4
    },
    'output': -1
})


# In[13]:


# cards is empty
tests.append({
    'input': {
        'cards': [],
        'query': 7
    },
    'output': -1
})


# In[14]:


tests.append({
    'input': {
        'cards': [8, 8, 6, 6, 6, 6, 6, 3, 2, 2, 2, 0, 0, 0],
        'query': 3
    },
    'output': 7
})


# In the case where `query` occurs multiple times in `cards`, we'll expect our function to return the first occurrence of `query`. 
# 
# While it may also be acceptable for the function to return any position where `query` occurs within the list, it would be slightly more difficult to test the function, as the output is non-deterministic.

# In[15]:


# query occurs multiple times
tests.append({
    'input': {
        'cards': [8, 8, 6, 6, 6, 6, 6, 6, 3, 2, 2, 2, 0, 0, 0],
        'query': 6
    },
    'output': 2
})


# Let's look at the full set of test cases we have created so far.

# In[16]:


tests


# Creating test cases beforehand allows us  to identify different variations and edge cases in advance so that can make sure to handle them while writing code. Sometimes, we may start out confused, but the solution will reveal itself as we try to come up with interesting test cases.

# ### 3. Come up with a correct solution for the problem. State it in plain English.
# 
# Our first goal should always be to come up with a _correct_ solution to the problem, which may necessarily be the most _efficient_ solution. The simplest or most obvious solution to a problem, which generally involves checking all possible answers is called the _brute force_ solution.
# 
# In this problem, coming up with a correct solution is quite easy: Bob can simply turn over cards in order one by one, till he find a card with the given number on it. Here's how we might implement it:
# 
# 1. Create a variable `position` with the value 0.
# 3. Check whether the number at index `position` in `card` equals `query`.
# 4. If it does, `position` is the answer and can be returned from the function
# 5. If not, increment the value of `position` by 1, and repeat steps 2 to 5 till we reach the last position.
# 6. If the number was not found, return `-1`.
# 
# > **Linear Search Algorithm**: An algorithm is simply a list of statements which can be converted into code and executed by a computer on different sets of inputs. This particular algorithm is called linear search, since it involves searching through a list in a linear fashion i.e. element after element.
# 
# 
# we should always try to express(speak or write) the algorithm in our own words before start coding.

# ### 4. Implement the solution and test it using example inputs. Fix bugs, if any.
# 
#  We are finally ready to implement our solution. All the work we've done so far will definitely come in handy, as we now exactly what we want our function to do, and we have an easy way of testing it on a variety of inputs.
# 
# Here's a first attempt at implementing the function

# In[17]:


def locate_card(cards, query):
    # Create a variable position with the value 0
    position = 0
    
    # Set up a loop for repetition
    while True:
        
        # Check if element at the current position matches the query
        if cards[position] == query:
            
            # Answer found! Return and exit..
            return position
        
        # Increment the position
        position += 1
        
        # Check if we have reached the end of the array
        if position == len(cards):
            
            # Number not found, return -1
            return -1


# In[18]:


test


# In[19]:


result = locate_card(test['input']['cards'], test['input']['query'])
result


# In[20]:


result == output


# The result matches the output. 
# 
# To help you test your functions easily the `jovian` Python library provides a helper function `evalute_test_case`. Apart from checking whether the function produces the expected result, it also displays the input, expected output, actual output from the function, and the execution time of the function.

# In[22]:


from jovian.pythondsa import evaluate_test_case


# 
# !pip install jovian --upgrade --quiet

# In[24]:


evaluate_test_case(locate_card,test)


# while it may seem like we have a working solution based on the above test, we can't be sure about it until we test the function with all the test cases. 
# 
# We can use the `evaluate_test_cases` (plural) function from the `jovian` library to test our function on all the test cases with a single line of code.

# In[25]:


from jovian.pythondsa import evaluate_test_cases


# In[26]:


evaluate_test_cases(locate_card, tests)


# our function encountered an error in the sixth test case. The error message suggests that we're trying to access an index outside the range of valid indices in the list. Looks like the list `cards` is empty in this case, and may be the root of the problem. 
# 
# Let's add some `print` statements within our function to print the inputs and the value of the `position` variable in each loop.

# In[27]:


def locate_card(cards, query):
    position = 0
    print('cards:',cards)
    print ('query:', query)
    
    while True:
        print('position:',position)
        
        if cards[position] == query:
            return position
        
        position += 1
        if position == len(cards):
            return -1


# In[29]:


cards6 = tests[6]['input']['cards']
query6 = tests[6]['input']['query']

locate_card(cards6, query6)


# Clearly, since `cards` is empty, it's not possible to access the element at index 0. To fix this, we can check whether we've reached the end of the array before trying to access an element from it. In fact, this can be terminating condition for the `while` loop itself.

# In[32]:


def locate_card(cards, query):
    position = 0
    while position < len(cards):
        if cards[position] == query:
            return position
        position += 1
    return -1


# In[34]:


tests[6]


# In[35]:


locate_card(cards6, query6)


# The result now matches the expected output. Do you now see the benefit of listing test cases beforehand? Without a good set of test cases, we may never have discovered this error in our function.
# 
# Let's verify that all the other test cases pass too

# In[36]:


evaluate_test_cases(locate_card, tests)


# Our code passes all the test cases. Of course, there might be some other edge cases we haven't thought of which may cause the function to fail. Can you think of any?
# 
# in real assesment , we can skip the step of implementing and testinh the brute force solution in the intreset of time . It's generally quite easy to figure out the complexity of the brute for solution from the plain English description.

# In[ ]:




