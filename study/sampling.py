import numpy as np

def bernoulli(n_samples,theta):
    return (np.random.random(n_samples) <= theta).astype(int)

if __name__ == '__main__':
    dataset_x = np.random.rand(10)
    print(dataset_x)
    
    theta = 0.7
    print(dataset_x <= theta) 
    
    print(bernoulli(10,theta=theta))
    print(bernoulli(100,theta=theta))