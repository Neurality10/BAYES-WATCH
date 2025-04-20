## ELL788 Computational Perception & Cognition

## Instructions
- Please answer Part-A and Part-B in separate answer books.
- Indicate the part clearly on top of the cover page of each answer book.

## PART - A

### 1. Short answer questions [4 x 1 = 4 marks]

1. Assume that a person is driving a car, while listening to music. The car's driver assistance system detects a pedestrian ahead on the road and estimates that the driver needs to apply his brakes within 10-15 ms to avoid an accident. Which of the following alarms will be most effective in this case?
    - Audio alarm, e.g. beeps from the speakers.
    - Visual alarm, e.g. flashing light on the panel.
    - Vibro-tactile alarm, e.g. a vibration on the seat.
Please justify your answer. Assume that the car has all the three provisions.

2. Assume that you have to distinguish some regions with five gray-shade colors (as illustrated in the adjoining diagram in three shades, white [255], gray [127] and black [0]). What color values will you use for achieving best perceptual distinction? Please justify your answer. [Hint: log 2 256 = 8]

3. a. Why does a haptic display require a high sampling rate?
   b. What is the importance of Surface Contact Point (SCP)?

4. a. Explain the principles behind MP3 and AAC perceptual audio coding schemes.
   b. What are the main differences between coding schemes used for music system and for telephony / conferencing?

5. a. What are the factors responsible for perception of size in human vision?
   b. Why does the left arrow in the adjoining figure generally appear to be shorter than the right one?

![Image 18](images/ell788_maj-img-1.png)

![Image 26](images/ell788_maj-img-2.png)

## PART - B

### 6. Long answer questions [5 x 3 = 15 marks]

6. Describe the key features of the massive modularity hypothesis of Cosmides and Tooby. How did this hypothesis come about on the basis of human performance on the Wason selection test? What do you think are the key weaknesses of this hypothesis? Is it compatible with Fodorean modularity: why or why not?

### 7. Long answer questions [5 x 3 = 15 marks]

7. In what way does ACT-R incorporate Bayesian inference? Explain using an example.

### 8. Long answer questions [5 x 3 = 15 marks]

8. In this course we have discussed two major types of cognitive modelling approaches: Bayesian and connectionist (neural network). In this question we will try to explore a simple example that can combine the two approaches. Consider a very basic neural network with just two inputs, x1 and x2, and a single neuron which computes y = σ(w0 + w1x1 + w2x2), where σ is the logistic sigmoid σ(a) = 1/(1 + e^-a). In other words, we are just referring to a standard logistic regression model. Further suppose that we have a data set with N points D = {(x11, x12, t1); (x21, x22, t2); ... (xN1, xN2, tN)}, where ti ∈ {0, 1} is the true output for data point i. As usual in logistic regression, we will interpret the model output y as the probability that the actual output is 1, i.e., p(ti = 1|x1i, x2i) = y(x1i, x2i).

   a. Draw the neural network, labeling all nodes and links.
   b. Normally neural networks are trained via backpropagation. Variants of this are also used in all the deep learning approaches we talked about. But here, we would like to use a Bayesian approach to infer the weights w = (w0, w1, w2). Write down Bayes' rule for the posterior distribution on the weights given the data, i.e., p(w|D).
   c. One term on the right hand side is the likelihood of the data. Given the model and data as described above, can you expand this term to explicitly write down the likelihood as a function of the model parameters and individual data points?
   d. The other term dependent on w is the prior. Can you give some reasonable choice for this?
   e. Having done parts (c) and (d), you get the posterior as a function of the given data points and the parameters. One can now maximise this (if not analytically then numerically or heuristically) to get the MAP estimates for the parameters w. How do you think these estimates relate to those that would have been obtained using standard backpropagation? Can you think of any advantage the Bayesian approach might have, especially in the context of the kinds of cognitive modelling tasks we've looked at?

### 9. Long answer questions [5 x 3 = 15 marks]

9. Consider the following three sentences:
   1. The daughter of the colonel who had a black dress left the party.
   2. The daughter of the colonel who had a black mustache left the party.
   3. The brother of the colonel who had a black mustache left the party.
   
   a. According to the race-based model of sentence processing, which of these three should be easiest to process? Explain why.
   b. The easiest to process sentence is actually problematic: can you say why? Does this indicate a flaw with race-based sentence processing?
