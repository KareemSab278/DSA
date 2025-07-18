# 217. Contains Duplicate
def containsDuplicate(nums):
        hashset = set(nums)
        if len(hashset) == len(nums):
            return False
        return True

# 242. Valid Anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        return Counter(s) == Counter(t)


# 1. Two Sum
class Solution(object):
    def twoSum(self, nums, target):
        hashmap = {}
        for i, val in enumerate(nums):
            diff = target - val
            if diff in hashmap:
                return [i, hashmap[diff]]
            hashmap[val] = i # hashmap[val] = i stores the index of val, so later if its pair (diff) shows up, you know where val was


# 49. Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = {}
        for word in strs:
            key = tuple(sorted(word))
            if key not in hashmap:
                hashmap[key] = []
            hashmap[key].append(word)
        return list(hashmap.values())
            
    
# top K Frequent Elements
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums) #get me the number of times a number appears
        common = count.most_common(k) #finds most common k elems

        result = []

        for num, _ in common:
            result.append(num)
        return result

# 271. Encode and Decode Strings
class Solution:

    def encode(self, strs: List[str]) -> str:
        output = ''
        for s in strs:
            output += str(len(s)) + '^' + s #o(n)
        return output

    def decode(self, s: str) -> List[str]:
        output = []
        i = 0
        while i < len(s):
            j = s.find('^', i) #always search from ith pos
            length = int(s[i:j]) #length is s[elem after i:elem before j] which is gonna be a str num so convert to int
            i = j + 1 #now the position starts from where ^ is
            output.append(s[i:i+length]) # now i add into output whatever is between s[i] and s[i+length] so like 10 to 14
            i += length # incrememnt i or else start from 0 again
        return output
    

# product of array except self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        result = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            result[i] = prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(len(nums)-1,-1,-1):
            result[i] *= postfix
            postfix *= nums[i]
        
        return result


# longest consecutive sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)

        longest = 0

        for n in nums:
            if (n-1) not in num_set:
                length = 0
                while(n + length) in num_set:
                    length += 1
                longest = max(length, longest)
        return longest
    

# valid palindrome
import re
class Solution:
    def isPalindrome(self, s: str) -> bool:
        #remove all non alphabetical letters and all to lowercase and remove all spaces too.
        cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
        return cleaned == ''.join(reversed(cleaned))
    

# two integer sum
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) -1

        while left < right:
            if numbers[right] + numbers[left] > target: #if  3+1 > 3
                right -= 1
            elif numbers[right] + numbers[left] < target: #if 1+1 = 2 < 3
                left += 1
            else:
                return [left+1, right+1]


# three sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            if i > 0 and a == nums[i - 1]:
                continue

            l = i+1
            r = len(nums) - 1

            while l < r:
                threeSum = a + nums[l] + nums[r]
                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
        return res
    

# container with most water
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height: return 0

        l = 0
        r = len(height)-1
        leftMax = height[l]
        rightMax = height[r]

        res = 0

        while l < r :
            if leftMax < rightMax:
                l += 1
                leftMax = max(leftMax, height[l])
                if leftMax - height[l] > 0:
                    res += leftMax - height[l]
            else:
                r -= 1
                rightMax = max(rightMax, height[r])
                if rightMax - height[r] > 0:
                    res += rightMax - height[r]
        return res
    
# valid parenthesis
class Solution:
    def isValid(self, s: str) -> bool:
        while '()' in s or '{}' in s or '[]' in s:
            s = s.replace('()', '')
            s = s.replace('{}', '')
            s = s.replace('[]', '')
        return s == ''


# minimum stack
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)

        if not self.min_stack: 
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))
        
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[len(self.stack)-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:

        stack = []
        for token in tokens:
            if token == '+':
                stack.append(stack.pop() + stack.pop())
            elif token == '-':
                a = stack.pop()
                b = stack.pop()
                stack.append(b-a)
            elif token == '/':
                a = stack.pop()
                b = stack.pop()
                stack.append(int(b / a))
            elif token == '*':
                stack.append(stack.pop() * stack.pop())
            else:
                stack.append(int(token))
        return int(stack[0])


# Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        stack = []
        output = []

        def backtrack(openN, closedN):
            if openN == closedN == n:
                output.append("".join(stack))
                return
            
            if openN < n:
                stack.append('(')
                backtrack(openN+1, closedN)
                stack.pop( )

            if closedN < openN:
                stack.append(')')
                backtrack(openN, closedN +1)
                stack.pop()
        
        backtrack(0,0)
        return output
    

# Daily temperatures
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []

        for i in range(len(temperatures)):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev = stack.pop()
                res[prev] = i - prev
            stack.append(i)

        return res


# Largest Rectangle In Histogram
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        def maxStack(heights):
            stack = []
            max_area = 0
            i = 0

            while i <= len(heights):
                h = heights[i] if i < len(heights) else 0
                if not stack or h >= heights[stack[-1]]:
                    stack.append(i)
                    i += 1
                else:
                    top = stack.pop()
                    width = i if not stack else i - stack[-1] - 1
                    area = heights[top] * width
                    max_area = max(max_area, area)

            return max_area

        return maxStack(heights)


# best time to buy and sell stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxProfit = 0
        left = 0
        right = left + 1

        while right < len(prices):
            if prices[left] > prices[right]:
                left = right
            elif prices[right] > prices[left]:
                maxProfit = max(maxProfit, (prices[right] - prices[left]))
            right += 1

        return maxProfit


# longest substring without repeating chars
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l = 0
        r = 0
        window = []
        maxLen = 0

        while r < len(s):
            if s[r] not in window:
                window.append(s[r])
                maxLen = max(maxLen, len(window))
                r+=1
            else:
                window.pop(0)
                l+=1
            
        return maxLen


# Longest Repeating Character Replacement
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0
        l = 0

        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            while (r- l +1) - max(count.values()) > k:
                count[s[l]] -= 1
                l += 1

            res = max(res, r -l +1)

        return res

