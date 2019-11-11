# Pandas -> Easy way of processing/manipulating tabular data

# Pandas -> load data
# Numpy -> Store as numpy array
# SK Learn -> Do ML stuff

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# CSV - CommaSeparatedValues
df = pd.read_csv("Pandas/PastHires.csv")
# Using pandas, with one line of code, we have created a dataframe of the data in the CSV
# df has now been converted to a PANDAS DATAFRAME

print(df.head()) # Returns first 5 rows of data

# We can also pass in an integer to head()
print(df.head(10)) # Returns the first 10 rows of data
print(df.head(1)) # Returns 1st row of data

# We can also look at the bottom/end of our data file
print(df.tail()) # Returns last 5 rows
print(df.tail(3)) # Returns last 3 rows

# Shape of data -> Dimensionality of dataframe
print(df.shape) # This will return the tuple (13,7) 13 rows x 7 columns
print(df.size) # Number of cells in dataframe (13*7=91)
print(len(df)) # 13 (number of rows in dataframes)

# Columns
print(df.columns) # Returns a list of columns (headers)

# Extracting single column from dataframe, let's say 'Hired'
print(df['Hired'])
# Extracts hired column
hired_column = df['Hired'] # This is a SERIES/1D ARRAY

# Extracted from an extracted column:
extracted_hired_column = df['Hired'][:5] # First 5 rows
print(extracted_hired_column)

extracted_hired_column = df['Hired'][5:] # From 5th row to end
print(extracted_hired_column)

# Extracting single values
extracted_value = df['Hired'][5] # value At index 5
print(extracted_value)

# Extracting more than 1 value
two_columns = df[['Years Experience','Hired']] # Column names
print(two_columns)

# we can also extract ranges of rows ex: [:5]
two_columns_w_range = df[['Years Experience','Hired']][:5] # First 5 rows
print(two_columns_w_range)

# Sorting values
print(df.sort_values(['Years Experience'])) # Returns sorted by years experience

# Value Counts - Gives count of each unique value
degree_counts = df['Level of Education'].value_counts()
print(degree_counts)

# Exercise
# Try extracting rows 5-10 of our dataframe, preserving only the previous employers and hired columns
df = df[['Previous employers','Hired']][5:11]
print(df)