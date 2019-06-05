#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:10:09 2019

@author: dmarr
"""
import os

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' build successfully!')
        return True
    else:
        print(path+' is already existing!')
        return False
 