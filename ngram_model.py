import nltk
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from copy import deepcopy

'''
read_blogs(blog_dir)
function to find all meaningful text from blog posts and return it all in one string
@:param blog_dir: path of files to be read
@:returns:a string containing all text from each file in the given directory
'''


def read_blogs(blog_dir):
    output_string = ""
    # store all files into a corpus using PlainTextCorpusReader
    file_corpus = nltk.corpus.PlaintextCorpusReader(blog_dir, ".*",
                                                    encoding="ISO-8859-1")  # odd encoding issue, had to switch to ISO-8859-1

    blog_names = file_corpus.fileids()
    ordered_blog_names = {}
    for blog_name in blog_names:  # loop through the file ids and store them in a new dictionary 
        split_name = blog_name.split(".")
        ordered_blog_names[int(split_name[0])] = blog_name  
    sorted_ids = sorted(ordered_blog_names.keys())  # sort ids in ascending order

    for id in sorted_ids:
        file_id = ordered_blog_names[id]
        file = file_corpus.open(file_id)
        post_marker = False  # boolean marker to tell the function if we are in a post or not

        for line in file.readlines():
            line = line.strip()  # strips each line of extra whitespace
            if "</post>" in line:
                post_marker = False

            if post_marker and len(line) > 0:  # if we are in between post markers, and it isn't an empty character, add to string
                output_string += line

            if "<post>" in line:
                post_marker = True

    return output_string


'''
count_words(text)
function that given a string, tokenizes it into sentences, then into words, 
takes a count of tokens and unique types in the string and stores 
1-grams, 2-grams and 3-grams in respective dictionaries
@:param text: string input
@:prints: number of types and number of tokens in given string
@:returns: uni,bi,trigram dictionaries
'''


def count_words(text):
    tokenized_text = sent_tokenize(text)  # tokenize whole text into sentence tokens
    # dictionary initializers for ngrams (defaultdict is used to eliminate if-else statements
    #            to check if value already exists in the dictionary, same reason for use of set() for the types)
    one_grams_dict = defaultdict(int)
    two_grams_dict = defaultdict(int)
    three_grams_dict = defaultdict(int)
    token_count = 0
    type_list = set()

    for sentence in tokenized_text:
        sentence = sentence.lower()  # changes sentence to lowercase
        tokenized_sentence = word_tokenize(sentence)  # tokenizes full sentence into word tokens
        tokenized_sentence.insert(0, '<s>')
        tokenized_sentence.append('</s>')
        token_count += len(tokenized_sentence)  # adds each length of sentence to token_count

        index = 0
        for word in tokenized_sentence:
            type_list.add(word)  # adds word into type_list if it hasn't appeared
            one_grams_dict[word] += 1
            if index < len(tokenized_sentence) - 1:  # checks to see if index is in bounds
                two_grams_dict[(word + " " + tokenized_sentence[index + 1])] += 1  # if in bounds, add word and next word to dict
            if index < len(tokenized_sentence) - 2:
                three_grams_dict[(word + " " + tokenized_sentence[index + 1] + " " + tokenized_sentence[index + 2])] += 1
            index += 1

    print("Number of tokens: " + str(token_count) + "\nNumber of types: " + str(len(type_list)))
    return one_grams_dict, two_grams_dict, three_grams_dict


'''
print_frequent_n_grams(unigrams, bigrams, trigrams, k)
function that prints the most frequently occurred uni, bi and trigrams in the corpus given their respective 
dictionaries
@:param unigrams: dictionary of all unigrams in training corpus
@:param bigrams: dictionary of all bigrams in training corpus
@:param trigrams: dictionary of all trigrams in training corpus
@:param k: desired amount of most frequent ngrams
@:prints: top k most frequent uni, bi and tri-grams
'''


def print_frequent_n_grams(unigrams, bigrams, trigrams, k):
    print("Top " + str(k) + " most frequent unigrams: " + str(nltk.FreqDist(unigrams).most_common(k))
          + "\nTop " + str(k) + " most frequent bigrams: " + str(nltk.FreqDist(bigrams).most_common(k))
          + "\nTop " + str(k) + " most frequent trigrams: " + str(nltk.FreqDist(trigrams).most_common(k)))


'''
predict(text, unigrams, bigrams, trigrams)
function to predict the next word of a given string using uni-gram, bi-gram and tri-gram language models. 
Laplace (Add-1) smoothing is applied to probability values
@:param text: string input
@:param unigrams: dictionary of all unigrams in training corpus 
@:param bigrams: dictionary of all bigrams in training corpus 
@:param trigrams: dictionary of all trigrams in training corpus 
@:prints: uni, bi and tri-gram predictions and their associated probabilities
'''


def predict(text, unigrams, bigrams, trigrams):
    # deep copies of each dictionary were needed as to not "over-smooth" the dict values with every call
    uni_dict, bi_dict, tri_dict = deepcopy(unigrams), deepcopy(bigrams), deepcopy(trigrams)
    words = text.split()

    bigram_matches = {key: value for key, value in bi_dict.items() if key.startswith(words[-1] + " ")}  # dictionary of all bigrams that start with the last word of inputted text
    trigram_matches = {key: value for key, value in tri_dict.items() if key.startswith(words[-2] + " " + words[-1] + " ")}  # dictionary of all trigrams that start with the last two words of inputted text

# Laplace smoothing for each dictionary
    for unigram in uni_dict:
        uni_dict[unigram] += 1

    for bigram in bi_dict:
        bi_dict[bigram] += 1

    for trigram in tri_dict:
        tri_dict[trigram] += 1

# all probabilities calculated using the Laplace method (adding 1 to each numerator and adding the type count of the corpus to the denominator)
    unigram_suggestion = nltk.FreqDist(uni_dict).max()  # highest frequency unigram in training corpus
    unigram_probability = uni_dict[unigram_suggestion] / sum(uni_dict.values())

    bigram_suggestion = nltk.FreqDist(bigram_matches).max()  # highest frequency bigram in training corpus
    bigram_probability = bi_dict[bigram_suggestion] / (uni_dict[words[-1]] + len(uni_dict.keys()))

    trigram_suggestion = nltk.FreqDist(trigram_matches).max()  # highest frequency trigram in training corpus
    trigram_probability = tri_dict[trigram_suggestion] / (bi_dict[words[-2] + " " + words[-1]] + len(uni_dict.keys()))
    print(bigram_suggestion)
    print("INPUT:" + text +
          "\n1-gram suggestion:", unigram_suggestion, ", with a probability of", '{:.4%}'.format(unigram_probability),
          "\n2-gram suggestion:", bigram_suggestion, ", with a probability of", '{:.4%}'.format(bigram_probability),
          "\n3-gram suggestion:", trigram_suggestion, ", with a probability of", '{:.4%}'.format(trigram_probability))


def main():
    blogs_dir_path = sys.argv[1]
    blog_text = read_blogs(blogs_dir_path)
    unigrams, bigrams, trigrams = count_words(blog_text)
    print_frequent_n_grams(unigrams, bigrams, trigrams, 5)
    sample_texts = ["the past few", "pass the salt and", "jump over", "in the"]
    for sample in sample_texts:
        predict(sample, unigrams, bigrams, trigrams)


if __name__ == "__main__":
    main()

