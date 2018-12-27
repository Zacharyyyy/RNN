# Introduction
* A recurrent neural network that can train a particular finite state machine.
# Structure of the RNN 
![](https://i.imgur.com/GJTnDk6.png)
* **word vector**: the into word, that has been transfored to word to vector.
* **state vector**: a vector that remember the current state.
* **tensor**: a 3 dimensional array that store the probabiity of going from one state to anothor on receiving a particular word. For example, tensor[i][j][k] is the probability of going from i-th state to k-th state on receiving j-th word.
* the output of tensor "h" is the probability vector of reaching the next state from the current state.![](https://i.imgur.com/bo1qi8Z.png)
* "i": from i-th state
* "c": on j-th word
* this is also called **matching**
* **normalize**: normalize the output vector of tensor, therefore it is a **probability**.
* **recurrent part**: take the output of normalization as the another part of input vector. This ia very intuitive, since the next state depends on the current state.
* **adder(add up)**: a vector that store the probability of each state being a terminal state.
# Training
* **Loss Function**: use quadratic loss function
* L(NN(w),y(w)) =(NN(w) − y(w))^2 
* L′(NN(w), y(w)) = 2*(NN(w) − y(w))
* **Transition function Match**
* gradient with respect to the state part of input(recurrent) ![](https://i.imgur.com/kQh5oVu.png)
* gradient with respect to tensor ![](https://i.imgur.com/6dyqrZW.png)
* **normalize**:
* gradient with respect to the current state![](https://i.imgur.com/A0h9sTv.png) 
* **delta tensor**:
* walk on the error surface according to the gradient of matching function with respect to itself.
* Adjusting weight
* **tensor**: for a sentence with |w|, delta_tensor has been calculated for |w| times. The value of tensor is adjusted with sum of these vector, before it move on to the next sentence.
* **adder**: adjest with the delta

# Example
## sentence_0
![](https://i.imgur.com/wN9cR4w.png)
* error rate 4.2167154180955916e-08
* terminal state: 3
* with this data set especially for learning plural grammar, the fsm learned that you need to add an 's'  and use 'are' for plural.
