# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:43:55 2020

@author: Deep Chokshi
"""
#Product of Largest two number
def max_pairwise_product_fast(n, numbers):
    max_index1 = -1
    for i in range(n):
        if max_index1 == -1 or numbers[i] > numbers[max_index1]:
            max_index1 = i

    max_index2 = -1
    for i in range(n):
        if i != max_index1 and (max_index2 == -1 or numbers[i] > numbers[max_index2]):
            max_index2 = i

    return numbers[max_index1] * numbers[max_index2]


if __name__ == '__main__':
    n = int(input())
    a = [int(x) for x in input().split()]
    assert (len(a) == n)

    print(max_pairwise_product_fast(n, a))
