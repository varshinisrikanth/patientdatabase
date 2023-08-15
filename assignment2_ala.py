import numpy as np

# Given dataset as a list of patient coordinates
dataset = [
    [76,126,38.0],
    [74,120,38.0],
    [72,118,37.5],
    [78,136,37.0],]
    


# Convert the dataset to a numpy array for easier calculations
data = np.array(dataset)

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# 1. Find the patient farthest from the rest
farthest_patient_idx = np.argmax(np.sum((data - np.mean(data, axis=0)) ** 2, axis=1))
farthest_patient = data[farthest_patient_idx]
print("Farthest patient:", farthest_patient)

# 2. Find the two nearest patients
distances = np.sqrt(np.sum((data[:, np.newaxis] - data) ** 2, axis=2))
np.fill_diagonal(distances, np.inf)
nearest_patient_idxs = np.unravel_index(np.argmin(distances), distances.shape)
nearest_patients = data[nearest_patient_idxs]
print("Nearest patients:", nearest_patients)

# 3. Create a new dummy patient and find the closest patient
dummy_patient = np.array([70,122,36])
closest_patient_idx = np.argmin(np.sqrt(np.sum((data -dummy_patient) ** 2, axis=1)))
closest_patient = data[closest_patient_idx]
print("Closest patient to the dummy:", closest_patient)
