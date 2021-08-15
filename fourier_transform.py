import numpy as np

# REFERENCES
#https://towardsdatascience.com/fast-fourier-transform-937926e591cb

#https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
#https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/


def show_M(N):
    """
    N: int 
    """

    n = np.arange(N)
    k = n.reshape((N,1))

    M  = k*n
    print("M:", M)  


def get_data(len):
    """
    len: int 
        lenght of data 
    """
    data = np.random.random(len)
    return data 

def get_circular_terms(N):
    """
    N: int 
    """

    terms =  np.exp(-1j *2*np.pi * np.arange(N)/N)

    return terms

def discrete_fourier_transform(data):
    """
    data: np.array 
        1 dimensional array
    """
    #len of data
    N =data.shape[0] 
    
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-1j * 2*np.pi * k * n/N)
   
    return np.dot(M,data)

def fast_fourier_transform(data):
    """
    data: np.array  
        data as 1D array
    return discrete fourier transform of data
    """

    # len of data
    N = data.shape[0]

    # Must be a power of 2
    assert   N % 2 == 0, 'len of data: {} must be a power of 2'.format(N)

    if N<= 2:
        return discrete_fourier_transform(data)

    else:
        data_even = fast_fourier_transform(data[::2])
        data_odd = fast_fourier_transform(data[1::2])
        terms = get_circular_terms(N)

        return np.concatenate(
            [
            data_even + terms[:N//2] * data_odd,
            data_even + terms[N//2:] * data_odd 
            ])
    

N = 4

X = get_data(N)
print("Data: ",X)

dt =  discrete_fourier_transform(X)
fdft = fast_fourier_transform(X)
dtnp = np.fft.fft(X)

print('DFT:',fdft)

print(np.allclose(dt,dtnp),
    np.allclose(fdft,dtnp))

print("")
show_M(N)



