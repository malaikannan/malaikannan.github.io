---
title: An Introduction to Tokenizers in Natural Language Processing
---
# Tokenizers

### _Co-authored by Tamil Arasan, Selvakumar Murugan and Malaikannan Sankarasubbu

In Natural Language Processing (NLP), one of the foundational steps is transforming human language into a format that computational models can understand. This is where tokenizers come into play. Tokenizers are specialized tools that break down text into smaller units called tokens, and convert these tokens into numerical data that models can process.

Imagine you have the sentence:

> Artificial intelligence is revolutionizing technology.

To a human, this sentence is clear and meaningful. But we do not
understand the whole sentence in one shot(okay may be you did, but I am
sure if I gave you a paragraph or a even better an essay, you will not
be able to understand them in one shot), but we make sense of parts of
it like words and then phrases and understand the whole sentence as a
composition of meanings from its parts. It is just how things work,
regardless whether we are trying to make a machine mimic our language
understanding or not. This has nothing to do with the reason ML models
or even computers in general work with numbers. It is purely how
language works and there is no going around it.

ML models like everything else we run on computers can only work with
numbers, and we need to transform the text into number or series of
numbers (since we have more than one word). We have a lot of freedom
when it comes to how we transform the text into numbers, and as always
with freedom comes complexity. But basically, tokenization as a whole is
a two step process. Finding all the words and assigning a unique
number - an ID to each token.

There are so many ways we can segment a sentence/paragraph into pieces
like phrases, words, sub-words or even individual characters.
Understanding why particular tokenization scheme is better requires a
grasp of how embeddings work. If you\'re familiar with NLP, you\'d ask
\"Why? Tokenization comes before the Embedding, right?\" Yes, you\'re
right, but NLP is paradoxical like that. Don\'t worry we will cover that
as we go.

# Background

Before we venture any further, lets understand the difference between
Neural networks and our typical computer programs. We all know by now
that for traditional computer programs, we write/translate the rules
into code by hand whereas, NNs learn the rules(mapping across input and
output) from data by the process called training. You see unlike in
normal programming style, where we have a plethora of data-structures
that can help with storing information in any shape or form we want,
along with algorithms that jump up and down, back and forth in a set of
instructions we call code, Neural Networks do not allow us to have all
sorts of control flow we\'d like. In Neural Networks, there is only one
direction the \"program\" can run, left to right.

Unlike in traditional programs where the we can feed a program with
input in complicated ways, in Neural Networks, there are only fixed
number of ways, we can feed and it is usually in the form of vectors
(fancy name for list of numbers) and the vectors are of fixed size (or
dimension more precisely). In most DNNs, input and output sizes are
fixed regardless of the problem it is trying to solve. For example, CNNs
the input (usually image) size and number of channels is fixed. In RNNs,
the embedding dimensions, input vocabulary size, number of output labels
(classification problem e.g: sentiment classification) and or output
vocabulary size (text generation problems e.g: QA, translation) are all
fixed. In Transformer networks even the sentence length is fixed. This
is not a bad thing, constraints like these enable the network to
compress and capture the necessary information.

Also note that there are only few tools to test \"equality\" or
\"relevance\" or \"correctness\" for things inside the network because
only things that dwell inside the network are vectors. Cosine similarity
and attention scores are popular. You can think of vectors as variables
that keep track of state inside neural network program. But unlike in
traditional programs where you can declare variables as you\'d like and
print them for troubleshooting, in networks the vector-variables are
only meaningful only at the boundaries of the layers(not entirely true)
within the networks.

Lets take a look at the simplest example to understand why just pulling
a vector from anywhere in the network will not be of any value for us.
In the following code, three functions perform the identical calculation
despite their code is slightly different. The unnecessarily
intentionally named variables `temp` and `growth_factor` need not be
created as exemplified by the first function, which directly embodies
the compound interest calculation formula, $A = P(1+\frac{R}{100})^{T}$.
When compared to `temp`, the variable `growth_factor` hold a more
meaningful interpretation - *represents how much the money will grow due
to compounding interest over time*. For more complicated formulae and
functions, we might create intermediate variables so that the code goes
easy on the eye, but they have no significance to the operation of the
function.

``` python
def compound_interest_1(P,R,T):
    A = P * (math.pow((1 + (R/100)),T))
    CI = A - P
    return CI

def compound_interest_2(P,R,T):
    temp = (1 + (R/100))
    A = P * (math.pow(temp, T))
    CI = A - P
    return CI

def compound_interest_3(P,R,T):
    growth_factor = (math.pow((1 + (R/100)),T))
    A = P * growth_factor
    CI = A - P
    return CI
```

Another example to illustrate from operations perspective. Clock
arithmetic. Lets assign numbers 0 through 7 to weekdays starting from
Sunday to Saturday.

**Table 1**

| Sun  | Mon  | Tue  | Wed  | Thu  | Fri  | Sat  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 1    | 2    | 3    | 4    | 5    | 6    |

> John Conway suggests, a mnemonic device for thinking of the days of
> the week as Noneday, Oneday, Twosday, Treblesday, Foursday, Fiveday,
> and Six-a-day.

So if you want to know what day it is 137 days from today if today is
say, Thursday (i.e. 4). We can do $(4+137) mod 7 => 1$ i.e Monday. As
you can see adding numbers(days) in clock arithmetic results in a
meaningful output. You can days together to get another day. Okay lets
ask the question can we multiply two days together? Is it is in anyway
meaningful to multiply days? Just because we can multiply any number
mathematically, is it useful to do so in our clock arithmetic?

All of this digression is to emphasize that the embedding is **deemed**
to capture the meaning of words, vector from the last layers is deemed
to capture the meaning of a sentence lets say. But when you take a
vector (just because you can) within the layers for instance, it does
not refer to any meaningful unit such as words or phrases and sentence
as we understand it.

# A little bit of history

If you\'re old enough, you might remember that before transformers
became standard paradigm in NLP, we had another one EEAP (Embed, Encode,
Attend, Predict). I am grossly oversimplifying here, but you can think
of it as follows,

Embedding

:   Captures the meaning of words A matrix of size $N \times D$, where

    -   $N$ is the size of the vocabulary, i.e unique number of words in
        the language
    -   $D$ is the dimension of embedding, vector corresponding to each
        word.
    
    Lookup the word-vector (embedding) for each word

Encoding
:   Find the meaning of a sentence, by using the meaning captured in
    embeddings of the constituent words with help of RNNs like LSTM, GRU
    or transformers like BERT, GPT that take the embeddings and produce
    vector(s) for whole the sequence.

Prediction
:   Depending upon the task at hand, either assigns a label to the input
    sentence, or generate another sentence word by word.

Attention
:   Helps with Prediction by focusing on what is important right now by
    drawing a probability distribution (normalized attention scores)
    over the all words. Words with high score are deemed important.

As you can see above, $N$ is the vocabulary size, i.e unique number of
words in the language. And handful of years ago, language usually meant
the corpus at hand (in order of few thousands of sentences) and datasets
like CNN/DailyMail were considered huge. There were clever tricks like
anonymizing named entities to force the ML models to focus on language
specific features like grammar instead of open world words like names of
Places, Presidents, Corporations and Countries, etc. Good times they
were! Point is, it is possible that the corpus you have in your
possession might not have all the words of the language. As we have
seen, the size of the Embedding must be fixed before training the
network. By good fortune if you stumble upon a new dataset and hence new
words, adding them to your model was not easy, because Embedding needs
to extend to accommodate this new (OOV) words and that requires
retraining of the whole network. OOV means Out Of the current model\'s
Vocabulary. And this is why simply segmenting the text on empty spaces
will not work.

With that background, lets dive in.

# Tokenization

Tokenization is the process of segmenting the text into individual
pieces (usually words) so that ML model can digest them. It is the very
first step in any NLP system and influences everything that follows. For
understanding impact of tokenization, we need to understand how
embeddings and sentence length influence the model. We will call
sentence length as sequence length from here on, because sentence is
understood to be sequence of words, and we will experiment with sequence
of different things not just words, which we will call tokens.

Tokens can be anything

-   Words - `"telephone" "booth" "is" "nearby" "the" "post" "office"`
-   Multiword Expressions (MWEs) -
    `"telephone booth" "is" "nearby" "the" "post office"`
-   Sub-words -
    `"tele" "#phone" "booth" "is" "near " "#by" "the" "post" "office"`
-   Characters - `"t" "e" "l" "e" "p" ... "c" "e"`

We know segmenting the text based on empty spaces will not work, because
the vocabulary will keep growing. What about punctuations? Surely they
will help with words
`don't, won't, aren't, o'clock, Wendy's, co-operation`{.verbatim} etc,
same reasoning applies here too. Moreover segmenting at punctuations
will create different problems, e.g: `I.S.R.O > I, S, R, O`{.verbatim}
which is not ideal.

# Objectives of Tokenization

The primary objectives of tokenization are:

Handling OOV
:   Tokenizers should be able to segment the text into pieces so that
    any word in the language whether it is in the dataset or not, any
    word we might conjure in foreseeable future, whether it is a
    technical/domain specific terminology that scientists might utter to
    sound intelligent or commonly used by everyone in day to day life.
    An ideal tokenizer should be able to deal with all and any of them.

Efficiency
:   Reducing the size (length) of the input text to make computation
    feasible and faster.

Meaningful Representation
:   Capturing the semantic essence of the text so that the model can
    learn effectively. Which we will discuss a bit later.

# Simple Tokenization Methods

Go through the code below, and see if you can make any inferences on the
table produced. It reads the book [The
Republic](https://www.gutenberg.org/cache/epub/1497/pg1497.txt) and
counts the tokens on character, word and sentence levels and also
indicated the number of unique tokens in the whole book.

## Code

``` {.python results="output raw" exports="both"}
from collections import Counter
from nltk.tokenize import sent_tokenize
with open('plato.txt') as f:
    text = f.read()

words = text.split()
sentences = sent_tokenize(text)

char_counter = Counter()
word_counter = Counter()
sent_counter = Counter()

char_counter.update(text)
word_counter.update(words)
sent_counter.update(sentences)

print('#+name: Vocabulary Size')
print('|Type|Vocabulary Size|Sequence Length|')
print(f'|Unique Characters|{len(char_counter)}|{len(text)}')
print(f'|Unique Words|{len(word_counter)}|{len(words)}')
print(f'|Unique Sentences|{len(sent_counter)}|{len(sentences)}')
```

**Table 2**

| Type              | Vocabulary Size | Sequence Length |
| ----------------- | --------------- | --------------- |
| Unique Characters | 115             | 1,213,712       |
| Unique Words      | 20,710          | 219,318         |
| Unique Sentences  | 7,777           | 8,714           |



## Study

Character-Level Tokenization

:   In this most elementary method, text is broken down into individual
    characters.

    *\"data\"* \> `"d" "a" "t" "a"`{.verbatim}

Word-Level Tokenization

:   This is the simplest and most used (before sub-word methods became
    popular) method of tokenization, where text is split into individual
    words based on spaces and punctuation. Still useful in some
    applications and as a pedagogical launch pad into other tokenization
    techniques.

    *\"Machine learning models require data.\"* \>
    `"Machine", "learning", "models", "require", "data", "."`{.verbatim}

Sentence-Level Tokenization

:   This approach segments text into sentences, which is useful for
    tasks like machine translation or text summarization. Sentence
    tokenization is not as popular as we\'d like it to be.

    *\"Tokenizers convert text. They are essential in NLP.\"* \>
    `"Tokenizers convert text.", "They are essential in NLP."`{.verbatim}

n-gram Tokenization

:   Instead of using sentences as a tokens, what if you could use
    phrases of fixed length. The following shows the n-grams for n=2,
    i.e 2-gram or bigram. Yes the `n`{.verbatim} in the n-grams stands
    for how many words are chosen. n-grams can also be built from
    characters instead of words, though not as useful as word level
    n-grams.

    *\"Data science is fun\"* \>
    `"Data science", "science is", "is fun"`{.verbatim}.





**Table 3**

| Tokenization | Advantages                             | Disadvantages                                        |
| ------------ | -------------------------------------- | ---------------------------------------------------- |
| Character    | Minimal vocabulary size                | Very long token sequences                            |
|              | Handles any possible input             | Require huge amount of compute                       |
| Word         | Easy to implement and understand       | Large vocabulary size                                |
|              | Preserves meaning of words             | Cannot cover the whole language                      |
| Sentence     | Preserves the context within sentences | Less granular; may miss important word-level details |
|              | Sentence-level semantics               | Sentence boundary detection is challenging           |

As you can see from the table, the vocabulary size and sequence length
have inverse correlation. The Neural networks requires that the tokens
should be present in many places and many times. That is how the
networks understand words. Remember when you don\'t know the meaning of
a word, you ask someone to use it in sentences? Same thing here, the
more sentences the token is present, the better the network can
understand it. But in case of sentence tokenization, you can see there
are as many tokens in its vocabulary as in the tokenized corpus. It is
safe to say that each token is occuring only once and that is not a
healthy diet for a network. This problem occurs in word-level
tokenization too but it is subtle, the out-of-vocabulary(OoV) problem.
To deal with OOV we need to stay between character level and word-level
tokens, enter \>\>\> sub-words \<\<\<.

# Advanced Tokenization Methods

Subword tokenization is an advanced tokenization technique that breaks
text into smaller units, smaller than words. It helps in handling rare
or unseen words by decomposing them into known subword units. Our hope
is that, the sub-words decomposed from text, can be used to compose new
unseen words and so act as the tokens for the unseen words. Common
algorithms include Byte Pair Encoding (BPE), WordPiece, SentencePiece.

*\"unhappiness\"* \> `"un", "happi", "ness"`{.verbatim}

BPE is originally a technique for compression of data. Repurposed to
compress text corpus by merging frequently occurring pairs of characters
or subwords. Think of it like what and how little number of unique
tokens you need to recreate the whole book when you are free to arrange
those tokens in a line as many time as you want.

Algorithm

:   1.  *Initialization*: Start with a list of characters (initial
        vocabulary) from the text(whole corpus).
    2.  *Frequency Counting*: Count all pair occurrences of consecutive
        characters/subwords.
    3.  *Pair Merging*: Find the most frequent pair and merge it into a
        single new subword.
    4.  *Update Text*: Replace all occurrences of the pair in the text
        with the new subword.
    5.  *Repeat*: Continue the process until reaching the desired
        vocabulary size or merging no longer provides significant
        compression.

Advantages

:   -   Reduces the vocabulary size significantly.
    -   Handles rare and complex words effectively.
    -   Balances between word-level and character-level tokenization.

Disadvantages

:   -   Tokens may not be meaningful standalone units.
    -   Slightly more complex to implement.

## Trained Tokenizers

WordPiece and SentencePiece tokenization methods are extensions of BPE
where the vocabulary is not merely created by assuming merging most
frequent pair. These variants evaluate whether the given merges were
useful or not by measuring how much each merge maximizes the likelihood
of the corpus. In simple words, lets take two vocabularies, before and
after the merges, and train two language models and the model trained on
vocabulary after the merges have lower perplexity(think loss) then we
assume that the merges were useful. And we need to repeat this every
time we make a merge. Not practical, and hence there some mathematical
tricks we use to make this more practical that we will discuss in a
future post.

The iterative merging process is the training of tokenizer and this
training is different training of actual models. There are python
libraries for training your own tokenizer, but when you\'re planning to
use a pretrained language model, it is better to stick with the
pretrained tokenizer associated with that model. In the following
section we see how to train a simple BPE tokenizer, SentencePiece
tokenizer and how to use BERT tokenizer that comes with huggingface\'s
`transformers`{.verbatim} library.

## Tokenization Techniques Used in Popular Language Models

### Byte Pair Encoding (BPE) in GPT Models

GPT models, such as GPT-2 and GPT-3, utilize Byte Pair Encoding (BPE)
for tokenization.

``` {.python results="output code" exports="both"}
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer =  Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                     vocab_size=30000)
files = ["plato.txt"]

tokenizer.train(files, trainer)
tokenizer.model.save('.', 'bpe_tokenizer')

output = tokenizer.encode("Tokenization is essential first step for any NLP model.")
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)
print("Length: ", len(output.ids))
```

``` python
Tokens: ['T', 'oken', 'ization', 'is', 'essential', 'first', 'step', 'for', 'any', 'N', 'L', 'P', 'model', '.']
Token IDs: [50, 6436, 2897, 127, 3532, 399, 1697, 184, 256, 44, 42, 46, 3017, 15]
Length:  14
```

### SentencePiece in T5

T5 models use a Unigram Language Model for tokenization, implemented via
the SentencePiece library. This approach treats tokenization as a
probabilistic model over all possible tokenizations.

``` python
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=plato.txt --model_prefix=unigram_tokenizer --vocab_size=3000 --model_type=unigram')
```

``` {.python results="output code" exports="both"}
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("unigram_tokenizer.model")
text = "Tokenization is essential first step for any NLP model."
pieces = sp.EncodeAsPieces(text)
ids = sp.EncodeAsIds(text)
print("Pieces:", pieces)
print("Piece IDs:", ids)
print("Length: ", len(ids))
```

``` python
Pieces: ['▁To', 'k', 'en', 'iz', 'ation', '▁is', '▁essential', '▁first', '▁step', '▁for', '▁any', '▁', 'N', 'L', 'P', '▁model', '.']
Piece IDs: [436, 191, 128, 931, 141, 11, 1945, 123, 962, 39, 65, 17, 499, 1054, 1441, 1925, 8]
Length:  17
```

### WordPiece Tokenization in BERT

``` {.python results="output code"}
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Tokenization is essential first step for any NLP model."
encoded_input = tokenizer(text, return_tensors='pt')

print("Tokens:", tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]))
print("Token IDs:", encoded_input['input_ids'][0].tolist())
print("Length: ", len(encoded_input['input_ids'][0].tolist()))
```

### Summary of Tokenization Methods

**Table 4**

| Method           | Length | Tokens                                                       |
| ---------------- | ------ | ------------------------------------------------------------ |
| BPE              | 14     | ['T', 'oken', 'ization', 'is', 'essential', 'first', 'step', 'for', 'any', 'N', 'L', 'P', 'model', '.'] |
| SentencePiece    | 17     | ['▁To', 'k', 'en', 'iz', 'ation', '▁is', '▁essential', '▁first', '▁step', '▁for', '▁any', '▁', 'N', 'L', 'P', '▁model', '.'] |
| WordPiece (BERT) | 12     | ['token', '##ization', 'is', 'essential', 'first', 'step', 'for', 'any', 'nl', '##p', 'model', '.'] |

Different tokenization methods give different results for the same input
sentence. As we add more data to the tokenizer training, the differences
between WordPiece and SentencePiece might decrease, but they will not
vanish, because of the difference in their training process.

**Table 5**

| Model  | Tokenization Method    | Library         | Key Features                             |
| ------ | ---------------------- | --------------- | ---------------------------------------- |
| *GPT*  | Byte Pair Encoding     | `tokenizers`    | Balances vocabulary size and granularity |
| *BERT* | WordPiece              | `transformers`  | Efficient vocabulary, handles morphology |
| *T5*   | Unigram Language Model | `sentencepiece` | Probabilistic, flexible across languages |

# Tokenization and Non English Languages

Tokenizing text is complex, especially when dealing with diverse
languages and scripts. Various challenges can impact the effectiveness
of tokenization.

## Tokenization Issues with Complex Languages: With a focus on Tamil

Tokenizing text in languages like Tamil presents unique challenges due
to their linguistic and script characteristics. Understanding these
challenges is essential for developing effective NLP applications that
handle Tamil text accurately.

### Challenges in Tokenizing Tamil Language

1.  1\. Agglutinative Morphology

    Tamil is an agglutinative language, meaning it forms words by
    concatenating morphemes (roots, suffixes, prefixes) to convey
    grammatical relationships and meanings. A single word may express
    what would be a full sentence in English.

    Impact on Tokenization

    :   -   Words can be very lengthy and contain many morphemes.
            -   போகமுடியாதவர்களுக்காவேயேதான்

2.  2\. Punarchi and Phonology

    Tamil specific rules on how two words can be combined and resulting
    word may not be phonologically identical to its parts. The
    phonological transformations can cause problems with TTS/STT systems
    too.

    Impact on Tokenization

    :   -   Surface forms of words may change when combined, making
            boundary detection challenging.
            -   மரம் + வேர் \> மரவேர்
            -   தமிழ் + இனிது \> தமிழினிது

3.  3\. Complex Script and Orthography

    Tamil alphabet representation in Unicode is suboptimal for
    everything except for standardized storage format. Even simple
    operations that are intuitive for native Tamil speaker, are harder
    to implement because of this. Techniques like BPE applied on Tamil
    text will break words at completely inappropriate points like
    cutting an uyirmei letter into consonant and diacritic resulting in
    meaningless output.

    தமிழ் \> த ம ி ழ, ்

### Strategies for Effective Tokenization of Tamil Text

1.  Language-Specific Tokenizers

    Train Tamil specific subword tokenizers with initial seed tokens
    prepared by better preprocessing techniques to avoid
    [*problem-3*]{.spurious-link
    target="*3. Complex Script and Orthography"} type cases. Use
    morphological analyzers to decompose words into root and affixes,
    aiding in understanding and processing complex word forms.

## Choosing the Right Tokenization Method

### Challenges in Tokenization

-   Ambiguity: Words can have multiple meanings, and tokenizers cannot
    capture context. Example: The word *\"lead\"* can be a verb or a
    noun.
-   Handling Special Characters and Emojis: Modern text often includes
    emojis, URLs, and hashtags, which require specialized handling.
-   Multilingual Texts: Tokenizing text that includes multiple languages
    or scripts adds complexity, necessitating adaptable tokenization
    strategies.

### Best Practices for Effective Tokenization

-   Understand Your Data: Analyze the text data to choose the most
    suitable tokenization method.
-   Consider the Task Requirements: Different NLP tasks may benefit from
    different tokenization granularities.
-   Use Pre-trained Tokenizers When Possible: Leveraging existing
    tokenizers associated with pre-trained models can save time and
    improve performance.
-   Normalize Text Before Tokenization: Cleaning and standardizing text