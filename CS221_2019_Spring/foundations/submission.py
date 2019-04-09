import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sorted(text.lower().split())[-1]
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sum(list((x - y) ** 2 for x,y in zip(loc1, loc2))) ** 0.5
    # END_YOUR_CODE

############################################################
# Problem 3c
def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    l_words = sentence.split()
    length = len(l_words)
    pairs = {}
    for i in range(len(l_words) - 1):
      if l_words[i] in pairs:
        pairs[l_words[i]].append(l_words[i + 1])
      else:
        pairs[l_words[i]] = [l_words[i + 1]]
    # use a queue to do search
    sentences = []
    results = set()
    for cur_word in pairs.keys():
      sentences.insert(0, [cur_word])
    while len(sentences) > 0:
      cur_word = sentences.pop()
      if len(cur_word) == length:
        results.add(' '.join(cur_word))
        continue
      if cur_word[-1] in pairs:
        for cur_nbr in pairs[cur_word[-1]]:
          sentences.insert(0, cur_word + [cur_nbr])
    return results
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return sum(v1[comp] * v2[comp] for comp in (set(v1.keys()) & set(v2.keys())))
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for comp in (set(v1.keys()) | set(v2.keys())):
      v1[comp] = v1[comp] + scale * v2[comp] 
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    itemcount = collections.Counter(text.lower().split())
    result = set()
    for word in itemcount:
      if itemcount[word] == 1:
        result.add(word)
    return result
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    #print(text)
    length = len(text)
    if length == 0: 
      return 0
    matrix = [[1 if i == j else (2 if i-j==1 and text[i]==text[j] else 1) for i in range(length)] for j in range(length)]
    for i in range(2, length):
      for j in range(length - i):
        if text[j] == text[j + i]:
          matrix[j][j+i] = 2 + matrix[j+1][j+i-1]
        else:
          if matrix[j+1][j+i] > matrix[j][j+i-1]:
            matrix[j][j+i] = matrix[j+1][j+i]
          else:
            matrix[j][j+i] = matrix[j][j+i-1]
    #print(matrix[0][length - 1])
    return matrix[0][length - 1]
    # END_YOUR_CODE
