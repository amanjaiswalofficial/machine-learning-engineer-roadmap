##### Workflow - initialization to prediction
1. View all columns
2. List all the numerical continuous columns, numerical discrete columns, categorical columns, ordinal
3. Finding columns having empty or null values
4. Finding type for all columns, useful for converting
5. Drawing observations based on data, like
   1. 16.0% people had both atleast one parent and one sibling on boat
6. Drawing conclusions based on 5
   1. Col X can be dropped due to high null values
   2. Col A is isn't directly contributing to the target, so can be dropped
7. Pivoting features for more corelation information, like
   1. People having A as value in Col X, had more B in Col B
8. Using visualization to find more corelation information
9. Data Wrangling like
   1.  Dropping not needed columns
   2.  Extracting required value from columns containing more detail than needed.
   3.  Replacing not so frequent values with rare and other such combinations
   4.  Converting categorical values into numerical ones using pandas
   5.  Filling missing continious numerical values
   6.  Adding features made from other features
   7.  Combining some numeric features which can have distinct combinations
   8.  Creating ranges as values for continuous numerical features
10. Do train test split
11. Identify the model required and create model
12. Do prediction