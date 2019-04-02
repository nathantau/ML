from numpy import random
print(random.seed(0)) # Makes random generation match across

totals = {20:0,30:0,40:0,50:0,60:0,70:0}
purchases = {20:0,30:0,40:0,50:0,60:0,70:0}

total_purchases = 0
for _ in range(100000):
  age_decade = random.choice([20,30,40,50,60,70])
  purchase_probability = float(age_decade)/100.0
  totals[age_decade] += 1
  if random.random() < purchase_probability:
    purchases[age_decade] += 1
    total_purchases += 1

print(f'Totals: {totals}') # People in each age group / Relatively equal
print(f'Purchases: {purchases}') # Purchases per age group / Higher as age group is higher
print(f'Total Purchases: {total_purchases} ')

# There is a dependence between age and purchases

# Conditional Probability
ProbEBarF = float(purchases[30]/totals[30])
print(f'P(E|F): {ProbEBarF}') # Rougly 30 %

ProbF = float(totals[30]/100000)
print(f'P(F): {ProbF}') # Roughly 16 %

ProbE = float(total_purchases/100000)
print(f'P(E): {ProbE}') # Roughly 45 % 

ProbEF = float(purchases[30]/100000)
print(f'P(E,F): {ProbEF}')