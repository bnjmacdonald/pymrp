# STAN model

data {
    int<lower=0> K;             // number of covariates
    int<lower=0> M;             // number of group-level covariates
    int<lower=0> N;             // number of observations
    int<lower=0> J;             // number of groups
    int<lower=1, upper=J> group[N];  // group of observation i
    int y[N];                   // response variable
    matrix[N, K] X;             // design matrix
    matrix[N, M] U;             // group-level design matrix

    # for prediction
    int<lower=0> N2;            // number of cells
    int<lower=1> group_new[N2];  // group of cell i
    matrix[N2, K] X_new;        // cell design matrix
    matrix[N2, M] U_new;            // group-level cell design matrix
}
parameters {
    # vector[J] a;                // group intercepts
    vector[J] a;                // varying group intercepts
    real a_fixed;                // fixed intercept
    vector[K] b;                // individual-level slopes
    vector[M] z;                // group-level slopes
    real<lower=0,upper=100> sigma_a;    // standard deviation of intercepts
    # real mu_a;                  // mean of intercepts
}
transformed parameters {
    vector[N] y_logit;
    
    for (i in 1:N) {
        # OLD: m[i] = a[group[i]] + U[i] * g;
        y_logit[i] = a_fixed + a[group[i]] + X[i] * b + U[i] * z;
    }
}
model {
    # mu_a ~ normal(0, {1:.4f});
    a ~ normal(0, sigma_a);
    b ~ normal(0, 100);
    b ~ normal(0, 10);
    z ~ normal(0, 10);
    y ~ bernoulli_logit(y_logit);
}
generated quantities {
    # vector[N2] y_pred_logits;
    vector[N2] y_pred;
    # vector[N] a_pred;
    
    for (i in 1:N2) {
        # OLD: m_new[i] = a[group_new[i]] + U_new[i] * g;
        y_pred[i] = bernoulli_rng(inv_logit(a[group_new[i]] + X_new[i] * b + U_new[i] * z));
    }
}




# OLD
# data {
#     int<lower=0> K;             // number of covariates
#     int<lower=0> M;             // number of group-level covariates
#     int<lower=0> N;             // number of observations
#     int<lower=0> J;             // number of groups
#     int<lower=1,upper=J> group[N];  // group of observation i
#     int y[N];                   // response variable
#     matrix[N, K] X;             // design matrix
#     matrix[J, M] U;             // group-level design matrix

#     # for prediction
#     int<lower=0> N2;            // number of cells
#     int<lower=1,upper=J> group_new[N2];  // group of cell i
#     matrix[N2, K] X_new;        // cell design matrix
#     matrix[J, M] U_new;            // group-level cell design matrix
# }
# parameters {
#     # vector[J] a;                // group intercepts
#     vector[K] b;                // slopes
#     vector[M] g;                // group-level slopes
#     # real<lower=0,upper=100> sigma_a;    // standard deviation of intercepts
#     # real mu_a;                  // mean of intercepts
# }
# transformed parameters {
#     vector[N] y_logit;
#     vector[J] a;              // varying group intercepts
    
#     for (j in 1:J) {
#         a[j] = U[j] * g;
#     }
#     for (i in 1:N) {
#         # OLD: m[i] = a[group[i]] + U[i] * g;
#         y_logit[i] = a[group[i]] + X[i] * b;
#     }
# }
# model {
#     # mu_a ~ normal({0:.4f}, {1:.4f});
#     # a ~ normal(mu_a, sigma_a);
#     b ~ normal(0, 10);
#     g ~ normal(0, 10);
#     y ~ bernoulli_logit(y_logit);
# }
# generated quantities {
#     # vector[N2] y_pred_logits;
#     vector[N2] y_pred;
#     vector[J] a_pred;
    
#     for (j in 1:J) {
#         a_pred[j] = U_new[j] * g;
#     }
#     for (i in 1:N2) {
#         # OLD: m_new[i] = a[group_new[i]] + U_new[i] * g;
#         y_pred[i] = bernoulli_rng(inv_logit(a_pred[group_new[i]] + X_new[i] * b));
#     }
# }