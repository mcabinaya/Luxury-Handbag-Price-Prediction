#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:54:56 2018

@author: abinaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/abinaya/USC/Studies/NLCI/Project/Data/clean_data.csv")

### inner material

median_inner_material = {}
for key in df["inner material"].unique():
     median_inner_material[key] =  df.loc[df["inner material"] == key, "price"].median()

plt.figure()
plt.bar(range(len(median_inner_material)), median_inner_material.values(), align='center')
plt.xticks(range(len(median_inner_material)), list(median_inner_material.keys()))
plt.title("Inner Material")

### hardware - metal type

median_hardware_metal_type = {}
for key in df["hardware - metal type"].unique():
     median_hardware_metal_type[key] =  df.loc[df["hardware - metal type"] == key, "price"].median()

plt.figure()
plt.bar(range(len(median_hardware_metal_type)), median_hardware_metal_type.values(), align='center')
plt.xticks(range(len(median_hardware_metal_type)), list(median_hardware_metal_type.keys()))
plt.title("Hardware - Metal Type")


### hardware - strap type

median_hardware_strap_type = {}
for key in df["hardware - strap type"].unique():
     median_hardware_strap_type[key] =  df.loc[df["hardware - strap type"] == key, "price"].median()

plt.figure()
plt.bar(range(len(median_hardware_strap_type)), median_hardware_strap_type.values(), align='center')
plt.xticks(range(len(median_hardware_strap_type)), list(median_hardware_strap_type.keys()))
plt.title("Hardware - Strap Type")
