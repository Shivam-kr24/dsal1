# for solving that question we signature of our function.
def locate_card(cards, query):  # i am taking locate_cards for identifying question,
    pass


# took example to describe test case in example
cards = [13, 11, 10, 7, 4, 3, 3, 0]
query = 7
output = 3
# we can test our function by passing the inputs into function and coming the rsult with the expected output.

result = locate_card(cards, query)
print(result)
result == output
 #we get output  none , because function has nothing .
 # we shall represent our test cases as dictionaries to make it easier to test .

    test = {
      'input': {
          'cards':[13, 11, 10, 7, 4, 3, 1],
          'query': 7
      },
      'output': 3
     }
locate_card(**test['input'])==test['output']



