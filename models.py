import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import root
from scipy.special import logsumexp

class ind_model:
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def infer_maxent_1d(self, mu, sigma, xmin=0.0, xmax=20.0, n_grid=2000):

        if not xmin < mu < xmax:
            raise ValueError(f'mu={mu} must satisfy {xmin} < mu < {xmax}')

        if sigma <= 0:
            raise ValueError('sigma must be positive')

        x = np.linspace(xmin, xmax, n_grid)
        target_second_moment = mu**2 + sigma**2

        def get_distribution(a, b):

            logp = a * x + b * x**2

            logp = logp - np.max(logp)

            p_unnormalized = np.exp(logp)

            Z = np.trapezoid(p_unnormalized, x)

            p = p_unnormalized / Z

            return p, Z

        def equations(params):

            a, b = params

            p, Z = get_distribution(a, b)

            mean_model = np.trapezoid(x * p, x)

            second_moment_model = np.trapezoid(x**2 * p, x)

            return [mean_model - mu, second_moment_model - target_second_moment]

        initial_guess = np.array([0.0, -1.0 / (2 * sigma**2)])

        result = root(equations, initial_guess)

        if not result.success:
            raise RuntimeError(f'Optimization failed: {result.message}')

        a, b = result.x

        p, Z = get_distribution(a, b)

        return x, p, a, b, Z
    
    
    def infer_maxent_all(self, xmin=0.0, xmax=20.0, n_grid=2000):

        mu_array = np.asarray(self.mu)
        sigma_array = np.asarray(self.sigma)

        if mu_array.shape != sigma_array.shape:
            raise ValueError('mu_array and sigma_array must have the same shape')

        N = len(mu_array)

        p_all = np.zeros((N, n_grid))

        a_array = np.zeros(N)
        b_array = np.zeros(N)
        Z_array = np.zeros(N)

        for i in range(N):

            x, p, a, b, Z = self.infer_maxent_1d(mu=mu_array[i], sigma=sigma_array[i], xmin=xmin, xmax=xmax, n_grid=n_grid)

            p_all[i] = p

            a_array[i] = a
            b_array[i] = b
            Z_array[i] = Z

        return x, p_all, a_array, b_array, Z_array
    
    def sample_from_pdf(self, x, p, n_samples):

        dx = np.diff(x)

        cdf = np.zeros_like(x)

        # trapezoidal integration
        cdf[1:] = np.cumsum(0.5 * (p[:-1] + p[1:]) * dx)

        # Normalize
        cdf = cdf / cdf[-1]

        # Inverse transform sampling
        random_numbers = np.random.random(n_samples)

        samples = np.interp(random_numbers, cdf, x)

        return samples
    
    def generate_maxent_patterns(self, n_trials):

        # Infer all marginal distributions
        x, p_all, a_array, b_array, Z_array = self.infer_maxent_all()

        N = len(self.mu)

        patterns = np.zeros((n_trials, N))

        # Independently sample each variable
        for i in range(N):

            patterns[:, i] = self.sample_from_pdf(x=x, p=p_all[i], n_samples=n_trials)

        return patterns, a_array, b_array, Z_array
    
    

class corr_model:
    
    def __init__(self, Data=None, xmin=0, xmax=20, n_states=40):
        
        # Data.shape = (N, n_trials)
        if Data is None:
            pass
        else:
            self.Data = Data
            
        self.N = self.Data.shape[0]
        
        # Discret State.
        self.states = np.linspace(xmin, xmax, n_states)
        
        self.n_states = n_states
        
        # Parameters
        self.h = np.zeros(self.N)
        
        self.g = np.zeros(self.N)
        
        self.J = np.zeros((self.N, self.N))
            
    def calculate_moments(self, samples):

        mean = np.mean(samples, axis=0)

        second_moment = (samples.T @ samples) / len(samples)

        return mean, second_moment
    
    def gibbs_sample(self, x):
        
        for i in range(self.N):
            
            # Effective field
            effective_field = (self.h[i] + np.dot(self.J[i], x) - self.J[i, i] * x[i])
            
            # Log probability for every possible state
            log_prob = (self.g[i] * self.states**2 + effective_field * self.states)
            
            # Numerical stability
            log_prob -= np.max(log_prob)
            
            prob = np.exp(log_prob)
            
            prob /= np.sum(prob)
            
            # Sample new x_i
            x[i] = np.random.choice(self.states, p=prob)
        
        return x
    
    def generate_samples(self, n_samples, burn_in=100, thinning=5):
        '''
        Generate samples from the model.
        '''
        
        x = np.random.choice(self.states, size=self.N)
        
        # Burn-in
        for _ in range(burn_in):
            x = self.gibbs_sample(x)
        
        samples = np.zeros((n_samples, self.N))
        
        for k in range(n_samples):
            
            for _ in range(thinning):
                x = self.gibbs_sample(x)
            
            samples[k] = x
        
        return samples
    
    def train_pairwise_maxent(self, n_iterations=200, learning_rate=1e-5, model_samples=500, burn_in=100, thinning=5):


        N, n_trials = self.Data.shape

        # Data moments
        mean_data = np.mean(self.Data, axis=1)

        second_moment_data = (self.Data @ self.Data.T) / n_trials

        for iteration in range(n_iterations):

            # Generate samples from current model
            samples = self.generate_samples(n_samples=model_samples, burn_in=burn_in, thinning=thinning)

            # Model moments
            mean_model, second_moment_model = (self.calculate_moments(samples))

            # Mean error
            delta_mean = (mean_data - mean_model)

            # Second moment error
            delta_second = (second_moment_data - second_moment_model)

            # Update h_i
            self.h += (learning_rate * delta_mean)

            # Update g_i
            self.g += (learning_rate * np.diag(delta_second))

            # Update J
            delta_J = (learning_rate * delta_second)

            # Do not update diagonal J
            np.fill_diagonal(delta_J, 0)

            self.J += delta_J

            # Enforce symmetry
            self.J = (self.J + self.J.T) / 2

            # Remove diagonal
            np.fill_diagonal(self.J, 0)

            if iteration % 10 == 0:

                mean_error = np.mean(np.abs(delta_mean))

                pair_error = np.mean(np.abs(delta_second))

                print(
                    f"Iteration {iteration:4d} | "
                    f"mean error = {mean_error:.6f} | "
                    f"second moment error = {pair_error:.6f}"
                )

        return self.h, self.g, self.J
    