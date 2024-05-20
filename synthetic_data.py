import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)

# Define the types of tea
tea_types = ['OOLONG', 'HITAM', 'HIJAU']

# Number of samples
num_samples = 100

# Create a DataFrame to hold the data
data = {
    'TEH': np.random.choice(tea_types, num_samples),
    'MQ3': np.random.uniform(400, 800, num_samples),
    'MQ4': np.random.uniform(500, 1000, num_samples),
    'MQ5': np.random.uniform(1500, 2500, num_samples),
    'MQ135': np.random.uniform(600, 1200, num_samples)
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_tea_data.csv', index=False)

print("Synthetic dataset created and saved to 'synthetic_tea_data.csv'.")
