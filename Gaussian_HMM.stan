data {
  
  int<lower = 0> N;
  int<lower = 1> Z;
  real y[N];
}

transformed data{
  
  real K = Z;
  real min_sigma = 1e-4;
  real real_bound = 1e4;
}

parameters {
  
  real mu[Z];
  
  real<lower=min_sigma> sigma[Z];
  
  simplex[Z] p_init;
  simplex[Z] p_transition[Z];
}

transformed parameters{

  real forward_logp;
  real alpha[N, Z];
  
  // Forward algorithm
  {
    real acc[Z];
    
    for (z in 1:Z){
      
      alpha[1, z] = normal_lpdf(y[1] | mu[z], sigma[z]) + log(p_init[z]);
      
    }
    
    for (t in 2:N){
      
      for (z in 1:Z){       // current state
        
        for (j in 1:Z){     // previous state
          
          acc[j] = alpha[t - 1, j] + log(p_transition[j, z]) + normal_lpdf(y[t] | mu[z], sigma[z]);
          
        }
        
        alpha[t, z] = log_sum_exp(acc);
        
      }
    }
    
    forward_logp = log_sum_exp(alpha[N]);
  }
}

model {
  
  // priors
  p_init ~ dirichlet(rep_vector(1/K, Z));
  
  for (z in 1:Z){
    
    p_transition[z] ~ dirichlet(rep_vector(1/K, Z));
    
  }
  
  mu ~ normal(0, 100^2);
  
  sigma ~ scaled_inv_chi_square(1, 0.05);
    
  target += forward_logp;
}

generated quantities {
  
  int<lower = 1, upper = Z> y_star[N];
  real y_hat[N];
  
  int back_ptr[N, Z] ;
  real best_logp[N, Z] ;
  
  real log_p_y_star;
  
  real acc[Z];
  simplex[Z] y_last_prob;
  
  int y_last;
  int y_next;
  
  real x_hat;
  
  {
    
    real best_total_logp;
    
    for (z in 1:Z) {
      
      best_logp[1, z] = normal_lpdf(y[1] | mu[z], sigma[z]) + log(p_init[z]);
    }
    
    for (t in 2:N) {
      
      for (z in 1:Z) {
        
        best_logp[t, z] = negative_infinity();
        
        for (j in 1:Z) {
          
          real logp;
          
          logp = best_logp[t - 1, j] + log(p_transition[j, z]) + normal_lpdf(y[t] | mu[z], sigma[z]);
          
          if (logp > best_logp[t, z]) {
            
            back_ptr[t, z] = j;
            best_logp[t, z] = logp;
            
          }
        }
      }
    }
    
    log_p_y_star = max(best_logp[N]);
    
    for (z in 1:Z) {
      
      if (best_logp[N, z] == log_p_y_star) {
        
        y_star[N] = z;
        
      }
    }
    
    for (t in 1:(N - 1)) {
      
      y_star[N - t] = back_ptr[N - t + 1, y_star[N - t + 1]];
      
    }
    
    for (t in 1:N){
      
      y_hat[t] = mu[y_star[t]];
      
    }
  }
  
  // one-step ahead prediction
  {
    for (i in 1:Z){
      
      acc[i] = alpha[N, i] - min(alpha[N]) + 1;
      
    }
    
    for (i in 1:Z){
      
      y_last_prob[i] = acc[i] / sum(acc);
    }
    
    y_last = categorical_rng(y_last_prob);
    y_next = categorical_rng(p_transition[y_last]);
    
    x_hat = normal_rng(mu[y_next], sigma[y_next]);
  }
  
}
