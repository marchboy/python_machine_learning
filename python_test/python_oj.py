# -*- coding = utf-8 -*-

#list_num = raw_input().split(' ')
# def devi():
#     a = int(list_num[0])
#     b = int(list_num[1])
#     if a % b == 0:
#         print("YES")
#     else:
#         print("NO")
#devi()

# num = input()
# lst = []
# def devision():
#     for i in xrange(num):
#         if num % (i + 1) == 0:
#             lst.append(i+1)
#     if lst == [1, num]:
#         print("Y")
#     else:
#         print("N")
# devision()

# n = input()
# def fibonacci(n):
#     if n <= 1:
#         return n
#     fibo = fibonacci(n - 1) + fibonacci(n - 2)
#     return fibo
# fibo = fibonacci(n)
# print(fibo)

# import numpy as np
# list = input().split(' ')
# num_
# matrix = np.zeros((int(num_list[0], int(num_list[1]))))

# -*- coding: utf-8 -*- 
from itertools import combinations

class Solution(object):
    # def __init__(self, nums, target):
    #     self.nums = nums
    #     self.target = target

    def twoSum(self, nums, target):
        for combins in combinations(nums,2):
            if sum(combins) == target:
                index_0 = nums.index(combins[0])
                index_1 = nums.index(combins[1])
                if index_0 != index_1:
                    return [index_0, index_1]

# class Solution(object):
#     """
#     :type nums: List[int]
#     :type target: int
#     :rtype: List[int]
#     """
#     def twoSum(self, nums, target):
#         length = len(nums)
#         for i in range(length):
#             for j in range(length - i - 1):
#                 j = i + j + 1
#                 if i == length - 1:
#                     break
#                 a_1 = nums[i]
#                 a_2 = nums[j]
#                 print(i, a_1)
#                 print(j, a_2)
#                 if a_1 + a_2 == target:
#                     res = [i, j]
#                     return res
# more pythonic
class Solution_0(object):
    def twoSum(self, nums, target):
        for i, val in enumerate(nums):
            diff = target - val
            if diff in nums:
                j = nums.index(diff)
                if i != j:
                    return [i, j]

    def twoSum_x(self, nums, target):
        hashed = {}
        for i in range(len(nums)):
            if target-nums[i] in hashed: return [hashed[target-nums[i]], i]
            hashed[nums[i]] = i

class Solution_1(object):
    def twoSum(self, nums, target):
        d = dict()
        for i, val in enumerate(nums):
            diff = target -val
            print('i=%d'%i, 'val=%d'%val, 'diff=%d'%diff)
            if diff in d:
                print('-'*15)
                print(d)
                return [d[diff], i]
            d[val] = i


if __name__ == '__main__':
    nums = [5,6,3,9,8,4,3,2]
    target = 6
    ss = Solution_1()
    a = ss.twoSum(nums, target)
    print(a)

count = 0
id(count)
def function():
    count = 1
    print(id(count))
    print(count)
    count = count + 1
    print(id(count))
    print(count)
id(count)
print(count)


# --------------------------------------------------------
# 从排序数组中删除重复项
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cache = set()
        for i in nums:
            cache.add(i)
        return list(cache)

class Solution_1(object):
    def removeDuplicates(self, nums):
        if len(nums) <= 1:
            return len(nums)
        i = 1
        index = 1
        while i < len(nums):
            if nums[i] != nums[i-1]:
                nums[index] = nums[i]
                index += 1
            i += 1
        return index

class Solution_2:  
    def removeDuplicates(self, nums):  
        """ 
        :type nums: List[int] 
        :rtype: int 
        """  
        i = 0  
        while i < len(nums)-1:  
            if nums[i] == nums[i+1]:  
                nums.remove(nums[i])  
            else:  
                i += 1  
        return len(nums) 


if __name__ == '__main__':
    nums = [1,2,3,4,4,5,6,6]
    s = Solution_2()
    s.removeDuplicates(nums)
    print(nums)


# ----------------------------------------------------
# 买卖股票的最佳时机 II
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        for i, val in enumerate(prices):
            for j, val_ in enumerate(prices[i+1:]):
                if val < val_:
                    buy_price = val
                    val = val_
                continue


if __name__ == '__main__':
    nums = [1,2,3,4,4,5,6,6]
    s = Solution()
    s.maxProfit(nums)


# ----------------------------------------------
# 旋转数组

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in range(k):
            out = nums.pop()
            nums.insert(0, out)

        return nums

    def rotate_1(self, nums, k):
        if k > len(nums):
            return None
        nums = nums[-k:] + nums[:-k]
        return nums

if __name__ == '__main__':
    nums = [1,2,3,4,5,6,7]
    s = Solution()
    s.rotate_1(nums,3)     


# -----------------------------------------
# 只出现一次的数字

class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        bak = set()
        for num in nums:
            if num in bak:
                bak.remove(num)
            else:
                bak.add(num)
        return list(bak)[0]
    def singleNumber_1(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for i in nums:
            # 异或运算
            a ^= i  
        return a
if __name__ == '__main__':
    nums = [4,1,2,1,2]
    s = Solution()
    s.singleNumber_1(nums)  

# ---------------------------------------
# 加一
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        # plusone = digits.pop() + 1
        # if plusone >= 10:  
        #     quotient = plusone // 10
        #     remainder = plusone % 10
        #     digits.extend([quotient,remainder])
        # else:
        #     digits.append(plusone)
        # return digits
        func = lambda x:str(x)
        plusone = int(''.join(map(func, digits))) +1

        # split_ = list(str(plusone))
        # res = list(map(lambda x: int(x), split_))
        res = list(map(int, str(plusone)))
        return res

    def plusOne_1(self, digits):
        res = 0
        l = len(digits) -1
        for i, num in enumerate(digits):
            res += num * 10 ** (l - i)
        res = list(map(int, str(res+1)))
 
        return res

if __name__ == '__main__':
    nums = [1,2,3]
    s = Solution()
    s.plusOne_1(nums)  

# -------------------------------------
# 移动零

class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # l = len(nums)
        # for i in range(l):
        #     num = nums[i]
        #     if num == 0:
        #         nums.remove(num)
        #         nums.append(num)
        #         print(nums,'-----')
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1

        return nums
if __name__ == '__main__':
    nums = [0,0,1]
    s = Solution()
    s.moveZeroes(nums)

# -------------------------
# 反转字符串
string = 'string'
string[::-1]


# 颠倒数字

class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        flag = 1 if x>= 0 else -1

        n = str(abs(x))[::-1]
        res = int(n) * flag
        return res if res.bit_length() < 32 else 0

# 字符串中的第一个唯一字符

class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """

        for i, val in enumerate(s):

            if val not in s[i+1:] and val not in s[:i]:
                return i

            if i+1 == len(s):
                return -1
        
        return -1

    def firstUniqChar_1(self, s):
        """
        :type s: str
        :rtype: int
        : 元素消除
        """
        unique = s
        while unique:
            if unique[0] in unique[1::]:
                unique = unique.replace(unique[0], "")
            else:
                return s.find(unique[0])
        return -1 


            
if __name__ == '__main__':
    s = Solution()
    s.firstUniqChar_1('')



# ------------------------------------
# 有效的字母异同位
# all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。
# 元素除了是 0、空、FALSE 外都算 TRUE。
# return set(s) == set(t) and all(s.count(i) == t.count(i) for i in set(s))

