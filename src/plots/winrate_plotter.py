import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'winners_for_70_no_mixup.csv'
data = pd.read_csv(csv_file_path)

data['Finetuning Steps'] = (data['Model Number'] + 1) * 100

frequency = data['Finetuning Steps'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
frequency.plot(kind='bar')
plt.title('Win Rate vs Number of Finetuning Steps')
plt.xlabel('Number of Finetuning Steps')
plt.ylabel('Win Rate (Frequency)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
