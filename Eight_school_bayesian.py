import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt

NUM_SCHOOLS = 8
NUM_RESULTS = 5000
NUM_BURNING_STEPS = 2500


def plot_treatments(data1, data2):
    fig, ax = plt.subplots()
    plt.bar(range(NUM_SCHOOLS), data1, yerr=data2)
    plt.title("8 Schools treatment effects")
    plt.xlabel("School")
    plt.ylabel("Treatment effect")
    fig.set_size_inches(10, 8)
    plt.show()


def model():
    m = tfp.distributions.JointDistributionSequential([
        tfp.distributions.Normal(
            loc=0., scale=10., name="avg_effect"),  # `mu` above
        tfp.distributions.Normal(
            loc=5., scale=1., name="avg_stddev"),  # `log(tau)` above
        tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(NUM_SCHOOLS),
                                                               scale=tf.ones(
            NUM_SCHOOLS),
            name="school_effects_standard"),  # `theta_prime`
            reinterpreted_batch_ndims=1),
        lambda school_effects_standard, avg_stddev, avg_effect: (
            tfp.distributions.Independent(tfp.distributions.Normal(loc=(avg_effect[..., tf.newaxis] +
                                                                        tf.exp(avg_stddev[..., tf.newaxis]) *
                                                                        school_effects_standard),  # `theta` above
                                                                   scale=treatment_stddevs),
                                          name="treatment_effects",  # `y` above
                                          reinterpreted_batch_ndims=1))
    ])
    return m


def target_log_prob_fn(avg_effect, avg_stddev, school_effects_standard):
    """Unnormalized target density as a function of states."""
    mod = model()
    return mod.log_prob((
        avg_effect, avg_stddev, school_effects_standard, treatment_effects))


@tf.function(autograph=False, jit_compile=True)
def do_sampling():
    return tfp.mcmc.sample_chain(
       num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNING_STEPS,
        current_state=[
            tf.zeros([], name='init_avg_effect'),
            tf.zeros([], name='init_avg_stddev'),
            tf.ones([NUM_SCHOOLS], name='init_school_effects_standard'),
        ],
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=0.4,
            num_leapfrog_steps=3))


if __name__ == '__main__':
    treatment_effects = np.array(
        [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)  # treatment effects
    treatment_stddevs = np.array(
        [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)  # treatment SE
    plot_treatments(treatment_effects, treatment_stddevs)

    states, kernel_results = do_sampling()
    avg_effect, avg_stddev, school_effects_standard = states
    
    school_effects_samples = (
        avg_effect[:, np.newaxis] +
        np.exp(avg_stddev)[:, np.newaxis] * school_effects_standard)

    num_accepted = np.sum(kernel_results.is_accepted)
    print('Acceptance rate: {}'.format(num_accepted /NUM_RESULTS))

    fig, axes = plt.subplots(8, 2, sharex='col', sharey='col')
    fig.set_size_inches(12, 10)
    for i in range(NUM_SCHOOLS):
        axes[i][0].plot(school_effects_samples[:,i].numpy())
        axes[i][0].title.set_text("School {} treatment effect chain".format(i))
        sns.kdeplot(school_effects_samples[:,i].numpy(), ax=axes[i][1], shade=True)
        axes[i][1].title.set_text("School {} treatment effect distribution".format(i))
    axes[NUM_SCHOOLS - 1][0].set_xlabel("Iteration")
    axes[NUM_SCHOOLS - 1][1].set_xlabel("School effect")
    fig.tight_layout()
    plt.show()

    print("E[avg_effect] = {}".format(np.mean(avg_effect)))
    print("E[avg_stddev] = {}".format(np.mean(avg_stddev)))
    print("E[school_effects_standard] =")
    print(np.mean(school_effects_standard[:, ]))
    print("E[school_effects] =")
    print(np.mean(school_effects_samples[:, ], axis=0))

    # Compute the 95% interval for school_effects
    school_effects_low = np.array([
        np.percentile(school_effects_samples[:, i], 2.5) for i in range(NUM_SCHOOLS)
    ])
    school_effects_med = np.array([
        np.percentile(school_effects_samples[:, i], 50) for i in range(NUM_SCHOOLS)
    ])
    school_effects_hi = np.array([
        np.percentile(school_effects_samples[:, i], 97.5)
        for i in range(NUM_SCHOOLS)
    ])

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.scatter(np.array(range(NUM_SCHOOLS)), school_effects_med, color='red', s=60)
    ax.scatter(
        np.array(range(NUM_SCHOOLS)) + 0.1, treatment_effects, color='blue', s=60)

    plt.plot([-0.2, 7.4], [np.mean(avg_effect),
                        np.mean(avg_effect)], 'k', linestyle='--')

    ax.errorbar(
        np.array(range(8)),
        school_effects_med,
        yerr=[
            school_effects_med - school_effects_low,
            school_effects_hi - school_effects_med
        ],
        fmt='none')

    ax.legend(('avg_effect', 'HMC', 'Observed effect'), fontsize=14)

    plt.xlabel('School')
    plt.ylabel('Treatment effect')
    plt.title('HMC estimated school treatment effects vs. observed data')
    fig.set_size_inches(10, 8)
    plt.show()
