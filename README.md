### Adaptive-SNN
How confident is the current prediction? (max softmax score)
How much does the top class beat the second class? (margin)
How "spread out" is the probability distribution? (entropy)
How many steps in a row has the prediction stayed the same? (stability)
What timestep is it? (position)
What was the confidence last step?###
1. Problem
Fixed-timestep SNN inference wastes compute on easy inputs. SEENN addresses this with a confidence-threshold early exit, but a single threshold ignores how the prediction behaves over time and can exit on a confident-but-wrong, unstable output.

2. Method
Train the SNN with a TET-style loss so every timestep's running-average output is a valid prediction target. At inference, instead of a fixed threshold, a small gradient-boosted classifier (the "gate") is trained on five per-timestep signals — confidence, margin, entropy, prediction run-length (stability), and timestep position — to directly predict "is the current prediction correct," and exits once that probability crosses a threshold.

3. Why it works
A single confidence threshold can't express patterns like "confident but still flip-flopping, keep going" or "lower confidence but stable for several steps, stop now." The gate learns these patterns from held-out validation data instead of relying on one hand-picked number.

4. Result
On CIFAR-10, at matched accuracy, the learned gate needs 6.5-15% fewer timesteps than a SEENN-I-style confidence threshold, and 30-40% fewer than fixed T=6, with less than 0.5-1.0 point of accuracy drop.
