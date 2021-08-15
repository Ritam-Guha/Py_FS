import numpy as np

def extract_info(file_path):
    # return weights, values as numpy arrays
    
    x = open(file_path, 'r')
    dimension = int(x.readline())
    
    weights = np.array(list(map(int, x.readline().split())))
    values = np.array(list(map(int, x.readline().split())))
    max_weight = int(x.readline())

    return dimension, weights, values, max_weight
    
    
def main():
    dimension, weights, values, max_weight = extract_info(file_path)
    print("======================================= %s =======================================" %(file_name))
    print("Number of objects:", dimension)
    print("Weights:", weights)
    print("Values:", values)
    print("Maximum allowable weight:", max_weight)
    print()
    
if __name__ == "__main__":
    file_base_path = 'Data/Knapsack/'
    file_names = ["ks_8a", "ks_8b", "ks_8c", "ks_8d", "ks_8e", "ks_12a", "ks_12b", "ks_12c", "ks_12d", "ks_12e", "ks_16a", "ks_16b", "ks_16c", "ks_16d", "ks_16e", "ks_20a", "ks_20b", "ks_20c", "ks_20d", "ks_20e", "ks_24a", "ks_24b", "ks_24c", "ks_24d", "ks_24e"]
    for file_name in file_names:
        file_path = file_base_path + file_name + ".txt"
        main()