import tensorflow as tf
from tensorflow_probability import distributions as tfd

data = [...]
weights = [...]
h = ...

f = lambda x: tfd.Independent(tfd.Normal(loc=x, scale=h))
n = data.shape[0].value

probs = weights / tf.reduce_sum(weights)

# https://astroviking.github.io/ba-thesis/tensorflowImplementation.html
kde = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=probs),
    components_distribution=f(data))

