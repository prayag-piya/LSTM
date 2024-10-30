import numpy as np
from typing import List, Tuple

class Tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.input = inputs
    
    def backward(self, dvalues):
        deriv = 1 - self.output ** 2
        self.dinputs = np.multiply(deriv, dvalues)

class Sigmoid:
    def forward(self, input):
        self.input = input
        self.output = np.clip(1 / (1 + np.exp(-input)), 1e-7, 1 - 1e-7)
    
    def backward(self, dvalue):
        sigmoid = self.output
        deriv = sigmoid * (1 - sigmoid)
        self.dinput = deriv * dvalue

class DenseLayer:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases

    def backward(self, dvalues):
        dvalues = dvalues.reshape(self.output.shape)
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinput = np.dot(dvalues, self.weights.T)

class LSTM:
    def __init__(self, n_input: int, n_neurons: int) -> None:
        self.n_input = n_input
        self.n_neurons = n_neurons

        # Weights and biases initialization
        self.Uf = 0.1 * np.random.randn(n_neurons, n_input) 
        self.bf = np.zeros((n_neurons, 1)) 
        self.Wf = 0.1 * np.random.randn(n_neurons, n_neurons)

        self.Ui = 0.1 * np.random.randn(n_neurons, n_input)
        self.bi = np.zeros((n_neurons, 1))
        self.Wi = 0.1 * np.random.randn(n_neurons, n_neurons)

        self.Uo = 0.1 * np.random.randn(n_neurons, n_input)
        self.bo = np.zeros((n_neurons, 1))
        self.Wo = 0.1 * np.random.randn(n_neurons, n_neurons)

        self.Ug = 0.1 * np.random.randn(n_neurons, n_input)
        self.bg = np.zeros((n_neurons, 1))
        self.Wg = 0.1 * np.random.randn(n_neurons, n_neurons)

    def forward(self, X_t):
        self.T = X_t.shape[0]
        self.X_t = X_t

        self.H = [np.zeros((self.n_neurons, 1)) for _ in range(self.T + 1)]
        self.C = [np.zeros((self.n_neurons, 1)) for _ in range(self.T + 1)]
        self.C_tilde = [np.zeros((self.n_neurons, 1)) for _ in range(self.T + 1)]

        self.F = [np.zeros((self.n_neurons, 1)) for _ in range(self.T)]
        self.O = [np.zeros((self.n_neurons, 1)) for _ in range(self.T)]
        self.I = [np.zeros((self.n_neurons, 1)) for _ in range(self.T)]

        self.dUf = np.zeros((self.n_neurons, self.n_input))
        self.dbf = np.zeros((self.n_neurons, 1))
        self.dWf = np.zeros((self.n_neurons, self.n_neurons))

        self.dUi = np.zeros((self.n_neurons, self.n_input))
        self.dWi = np.zeros((self.n_neurons, self.n_neurons))
        self.dbi = np.zeros((self.n_neurons, 1))
        
        self.dUo = np.zeros((self.n_neurons, self.n_input))
        self.dbo = np.zeros((self.n_neurons, 1))
        self.dWo = np.zeros((self.n_neurons, self.n_neurons))

        self.dUg = np.zeros((self.n_neurons, self.n_input))
        self.dbg = np.zeros((self.n_neurons, 1))
        self.dWg = np.zeros((self.n_neurons, self.n_neurons))

        self.Sigmf = [Sigmoid() for _ in range(self.T)]
        self.Sigmi = [Sigmoid() for _ in range(self.T)]
        self.Sigmo = [Sigmoid() for _ in range(self.T)]
        self.Tanh1 = [Tanh() for _ in range(self.T)]
        self.Tanh2 = [Tanh() for _ in range(self.T)]

        ht = self.H[0]
        ct = self.C[0]

        # LSTM Cell Call
        self.H, self.C, self.F, self.O, self.I, self.C_tilde = self.LSTMCELL(X_t, ht, ct)

    def LSTMCELL(self, X_t, ht, ct):
        for t in range(self.T):
            xt = X_t[t].reshape(-1, 1)

            # Forget gate
            outf = np.dot(self.Uf, xt) + np.dot(self.Wf, ht) + self.bf
            self.Sigmf[t].forward(outf)
            ft = self.Sigmf[t].output

            # Input gate
            outi = np.dot(self.Ui, xt) + np.dot(self.Wi, ht) + self.bi
            self.Sigmi[t].forward(outi) 
            it = self.Sigmi[t].output

            # Output gate
            outo = np.dot(self.Uo, xt) + np.dot(self.Wo, ht) + self.bo
            self.Sigmo[t].forward(outo)  
            ot = self.Sigmo[t].output

            # C tilde
            outct_tilde = np.dot(self.Ug, xt) + np.dot(self.Wg, ht) + self.bg
            self.Tanh1[t].forward(outct_tilde)
            ct_tilde = self.Tanh1[t].output

            ct = ft * ct + it * ct_tilde

            self.Tanh2[t].forward(ct)
            ht = self.Tanh2[t].output * ot

            self.H[t + 1] = ht
            self.C[t + 1] = ct
            self.C_tilde[t] = ct_tilde
            self.F[t] = ft
            self.O[t] = ot
            self.I[t] = it

        return self.H, self.C, self.F, self.O, self.I, self.C_tilde

    def backward(self, dvalues):
        dht = dvalues[-1].reshape(self.n_neurons, 1)

        for t in reversed(range(self.T)):
            xt = self.X_t[t].reshape(-1, 1)

            self.Tanh2[t].backward(dht)
            dtanh2 = self.Tanh2[t].dinputs

            dhtdtanh = self.O[t] * dtanh2

            dctdft = dhtdtanh * self.C[t - 1]
            dctdit = dhtdtanh * self.C_tilde[t]
            dctct_tilde = dhtdtanh * self.I[t]

            self.Tanh1[t].backward(dctct_tilde)
            dtanh1 = self.Tanh1[t].dinputs

            self.Sigmf[t].backward(dctdft)
            dsigmf = self.Sigmf[t].dinput

            self.Sigmi[t].backward(dctdit)
            dsigmi = self.Sigmi[t].dinput

            self.Sigmo[t].backward(dht)
            dsigmo = self.Sigmo[t].dinput

            self.dUf += np.dot(dsigmf, xt.T)
            self.dWf += np.dot(dsigmf, self.H[t-1].T) if t > 0 else 0 
            self.dbf += dsigmf

            self.dUi += np.dot(dsigmi, xt.T)
            self.dWi += np.dot(dsigmi, self.H[t-1].T) if t > 0 else 0 
            self.dbi += dsigmi

            self.dUo += np.dot(dsigmo, xt.T)
            self.dWo += np.dot(dsigmo, self.H[t-1].T) if t > 0 else 0 
            self.dbo += dsigmo

            self.dUg += np.dot(dtanh1, xt.T)
            self.dWg += np.dot(dtanh1, self.H[t-1].T) if t > 0 else 0 
            self.dbg += dtanh1

            dht = (np.dot(self.Wf.T, dsigmf) + np.dot(self.Wi.T, dsigmi) +
                   np.dot(self.Wo.T, dsigmo) + np.dot(self.Wg.T, dtanh1) +
                   (dvalues[t-1].reshape(self.n_neurons, 1) if t > 0 else 0))
        self.H = self.H 