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


