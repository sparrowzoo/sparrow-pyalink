#!/usr/bin/env python3
# -*- coding: utf-8 -*-

classmates = ['Michael', 'Bob', 'Tracy']
classmates.append("Harry")
classmates.insert(4,"Sparrow")
classmates.append("test")
classmates.remove("test")
classmates.pop()
classmates.pop(2)
print('classmates =', classmates)
print('len(classmates) =', len(classmates))
for item in classmates:
    print(item)
classmates[1]="Modified Bob"
print('classmates =', classmates)

print("sorted list=",classmates)

print("temp sort list=",sorted(classmates))
classmates.reverse()
print("reverse list=",classmates)
classmates.reverse()
print("reverse list=",classmates)
