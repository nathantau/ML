# Practice
from numpy import random
print(random.seed(0)) # Makes random generation match across

totals = {20:0,30:0,40:0,50:0,60:0,70:0}
purchases = {20:0,30:0,40:0,50:0,60:0,70:0}

total_purchases = 0
for _ in range(100000):
  age_decade = random.choice([20,30,40,50,60,70])
  # independent purchase probability
  purchase_probability = random.choice([0.1,0.2,0,3,0.4,0.5,0.6,0.7,0.8])
  totals[age_decade] += 1
  if random.random() < purchase_probability:
    purchases[age_decade] += 1
    total_purchases += 1

print(f'Totals: {totals}') # People in each age group / Relatively equal
print(f'Purchases: {purchases}') # Purchases per age group / Higher as age group is higher
print(f'Total Purchases: {total_purchases}')

# Condition that you bought something if you were 30
probEBarF = float(purchases[30]/totals[30])
print(probEBarF)

# Condition that you bought something and you were 30
probEF = float(purchases[30]/100000)
print(probEF)

probE = float(total_purchases/100000)
print(probE)

probF = float(totals[30]/100000)
print(probF)
