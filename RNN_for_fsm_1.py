import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



DICTIONARY = ['i', 'he', 'she', 'they', '','am', 'is', 'are', 'boy', 'girl', 'boys', 'girls','a', 'an',\
'it', 'dog', 'dogs', 'cat', 'cats', 'computer', 'computers', 'cup', 'cups', 'cake', 'cakes', \
'use', 'uses', 'using', 'eat', 'eats', 'eating'
]

NUMBER_OF_POSITIONS = 5
START_POSITION = np.zeros(NUMBER_OF_POSITIONS)
START_POSITION[0] = 1
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
TenzToAdd = 2.0
EPS = 0.000001
NU = 0.7
NU_ADDER = 0.3

class NeuralNetwork:

    tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])  # probability of going from state A to state B with character C
    adder = np.zeros(NUMBER_OF_POSITIONS)                                           #probability of of that i-state is terminal

    def __init__(self):
        self.tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        for fr in range(NUMBER_OF_POSITIONS):
            for ch in range(NUMBER_OF_CHARS):
                z = np.random.rand(NUMBER_OF_POSITIONS)
                z = normalize(z)
                for to in range(NUMBER_OF_POSITIONS):
                    self.tensor[to][ch][fr] = z[to]
        self.adder = np.zeros(NUMBER_OF_POSITIONS)
        for to in range(NUMBER_OF_POSITIONS):
            self.adder[to] = 1.0 * (to + 1) / (NUMBER_OF_POSITIONS)


    def check(self, word):
        curr_pos = START_POSITION
        words = word.split(" ")
        for k in range(len(words)):
            curr_word = char_to_vector(words[k])
            curr_pos = match(self, curr_word, curr_pos)
            curr_pos = normalize(curr_pos)
        return lastsum(self, curr_pos)

    def train_online(self, dataset):
        times = 0
        average_error = 1.0
        epoch_number = 0
        n = len(dataset)
        tests_size = int(PERCENT_OF_TESTS * n)
        while (average_error > EPS):
            random.shuffle(dataset)
            cases_left = len(dataset)
            epoch_number += 1
            print 'Epoch #' + str(epoch_number)
            while(cases_left > tests_size):
                self.train(dataset[cases_left - 1][0], dataset[cases_left - 1][1])
                cases_left -= 1
            average_error = 0.0
            for i in range(cases_left):
                average_error += cost_function(dataset[i][1], self.check(dataset[i][0]))
            average_error /= cases_left
            plt.plot(times, average_error, 'bo')
            times += 1
            print "Average error: " + str(average_error)

    def train(self, word, exp):
        cut_v = np.vectorize(cut)
        words = word.split(' ')
        word_length = len(words)

        positions = np.zeros([word_length + 1, NUMBER_OF_POSITIONS])
        before_normalize = np.zeros([word_length, NUMBER_OF_POSITIONS])				    # probability of going to each state, after ith word
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])# probability of going from state A to state B with character C
        d_adder = np.zeros(NUMBER_OF_POSITIONS)                                         # probability of of that i-state is terminal
        positions[0] = START_POSITION

        # apply the RNN to the input word
        for k in range(word_length):
            curr_word = char_to_vector(words[k])
            before_normalize[k] = match(self, curr_word, positions[k])      # matchs the probability of reaching each state with the given character and state
            positions[k + 1] = normalize(before_normalize[k])				# normalization, so that positions adds up to one, therefore it is a probability
        answer = lastsum(self, positions[-1])						        # adder dot final states possibility
        error = cost_function(exp, answer)

        # adjustind adder and tensor
        gradient = answer - exp			# difference between real answer and our answer
        d_adder += lastsum_derrivative_adder(self, gradient, positions[-1])		# gradient dot (final position multi adder)
        gradient = lastsum_derrivative(self, gradient)					# gradeint  = np.dot(gradient, self.adder)
        first_gradient = sum(abs(gradient))         # sum(adder dot gradient)

        # calculate the W delta tensor and add them up
        for k in range(word_length - 1, -1, -1):
            curr_grad = sum(abs(gradient))
            if (curr_grad < 0.001):
                koef = 1.0
            else:
                koef = first_gradient / sum(abs(gradient))
            gradient *= koef                    # gradient = koef * np.dot(gradient, self.adder); adjust gradient a bit
            curr_word = char_to_vector(words[k])
            gradient = normalize_derrivative(gradient, before_normalize[k]) # derrivative of normalize function times gradient
            d_tensor += match_derrivative_tensor(gradient, curr_word, positions[k]) # gradient of maych with respect to tensor, times gradient
            gradient = match_derrivative(self, gradient, curr_word) # derrivative of match function with respect to position, times gradient
        
        # adjust tensor with the delta tensor calculated
        d_tensor /= word_length
        self.tensor = cut_v(self.tensor - NU * d_tensor)
        self.adder = cut_v(self.adder - NU * NU_ADDER * d_adder)
        return error

    def get_automaton(self):
        for j in range(NUMBER_OF_POSITIONS):
            for i in range(NUMBER_OF_CHARS):
                max_ind = 0
                for k in range(1, NUMBER_OF_POSITIONS):
                    if (nn.tensor[k][i][j] > nn.tensor[max_ind][i][j]):
                        max_ind = k
                print str(j) + "--" + str(DICTIONARY[i]) + '-->' + str(max_ind)
        for k in range(NUMBER_OF_POSITIONS):
            if (nn.adder[k] > 0.5):
                print str(k) + " is terminal"

def cost_function(exp, res):
    return (res - exp) ** 2

# match(neural net, character, state), matchs the probability of reaching each state with the given character and state
def match(nn, ch, pos):
    new_pos = np.zeros(NUMBER_OF_POSITIONS)
    for k in range(NUMBER_OF_POSITIONS):
        for i in range (NUMBER_OF_CHARS):
            for j in range (NUMBER_OF_POSITIONS):
                new_pos[k] += nn.tensor[k][i][j] * ch[i] * pos[j]
    return new_pos

# gradient of match function with respect to positions
def match_derrivative(nn, dz, ch):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        for k in range(NUMBER_OF_POSITIONS):
            for j in range (NUMBER_OF_CHARS):
                derrivative[k][i] += nn.tensor[k][j][i] * ch[j]
    return np.dot(dz, derrivative) 

# gradient of match function with respect to tensor
def match_derrivative_tensor(dz, ch, pos):
    sample_matrix = np.zeros([NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_CHARS):
        for j in range(NUMBER_OF_POSITIONS):
            sample_matrix[i][j] = ch[i] * pos[j]            # gradient with respect to tensor
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        derrivative[i] = dz[i] * sample_matrix              # result times the error
    return derrivative

def normalize(t):
    return t / np.sum(t)

# gradient of normalizing function with respect to states
def normalize_derrivative(dz, inp):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    sum = np.sum(inp)           # sum(before_normalize[character]), the sum of all states
    for i in range(NUMBER_OF_POSITIONS):
        for j in range(NUMBER_OF_POSITIONS):
            if(i == j):
                derrivative[i][j] = (sum - inp[i]) / (sum ** 2)
            else:
                derrivative[i][j] = -inp[i] / (sum ** 2)
    return np.dot(dz, derrivative)              # gradient * derrivative

def lastsum(nn, x):
    return np.dot(nn.adder, x)

def lastsum_derrivative(nn, dz):
    return np.dot(dz, nn.adder)

def lastsum_derrivative_adder(nn, dz, inp):
    derrivative = np.multiply(inp, nn.adder) #last state times the adder vector
    return np.dot(dz, derrivative) #difference times derrivative

def char_to_vector(ch):
    index = DICTIONARY.index(ch.lower())
    vec = np.zeros(NUMBER_OF_CHARS)
    vec[index] = 1.0
    return vec

def cut(x):
    if (x > 1.0):
        return 1.0
    if (x < 0.0):
        return 0.0
    return x


f = open('sentence_0.txt', 'r')
dataset = []
for line in f:
    arr = line.split('\\')
    isOk = 1.0
    if arr[-1][0] == '0':
        isOk = 0.0
    dataset.append([arr[0], isOk])


nn = NeuralNetwork()
nn.train_online(dataset)
nn.get_automaton()

plt.xlabel('Times of view')
plt.ylabel('error rate')
plt.show()
