import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf


def happiness_model(weather_prob, weathe_to_happiness_probs):
    weather = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Bernoulli(probs=weather_prob, name="weather")
    )
    happiness = yield tfp.distributions.Bernoulli(probs=weathe_to_happiness_probs[weather], name="happiness")


def neg_log_likelihood(): return -tf.reduce_sum(model_joint_fit.log_prob(dataset))


if __name__ == '__main__':
    theta_weather = tf.constant(0.8)
    theta_happiness = tf.constant([0.7, 0.9])
    model_joint_original = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: happiness_model(theta_weather, theta_happiness),
    )
    dataset = model_joint_original.sample(100)
    
    theta_weather_fit = tfp.util.TransformedVariable(
        0.5, bijector=tfp.bijectors.SoftClip(low=0.0, high=1.0), name='theta_weather_fit')
    theta_happiness_fit = tfp.util.TransformedVariable(
        [0.5, 0.5], bijector=tfp.bijectors.SoftClip(low=0.0, high=1.0), name='theta_happiness_fit')
    model_joint_fit = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: happiness_model(theta_weather_fit, theta_happiness_fit),
    )

    print(dataset)
    print(model_joint_fit.log_prob(dataset))
    print(tfp.math.minimize(loss_fn=neg_log_likelihood,
                            optimizer=tf.optimizers.Adam(0.01), num_steps=1000))
    print(theta_weather_fit, theta_happiness_fit)
    print(theta_weather, theta_happiness)
