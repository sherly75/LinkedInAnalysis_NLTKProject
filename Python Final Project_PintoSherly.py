import pdfplumber,os, nltk
from textblob import TextBlob
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
import matplotlib.pyplot as plt


#in texts[list], 0-3 are CEO, 4-7 are Current students,
# 8-11 are Juniors level, 12-15 are Recent Graduates, 16-19 are Senior level

def get_pdf_text(filename):

    pdf = pdfplumber.open(DATA_DIR+os.sep+filename)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    pdf.close()
    return text




DATA_DIR = "Python Project"
from os import listdir
from os.path import isfile, join
allfiles = [f for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]
texts=[]
for file in allfiles[:]:
    texts.append(get_pdf_text(file))
    #print(f'file {len(texts)} of {len(allfiles)}')






#FREQUENCY
wordlist=[]
totalwords=[]
for t in texts:
    wordings=(t.split())
    wordlist.append(wordings)
    total=len(wordings)
    #print(total)
    totalwords.append(total)

#print(totalwords[n])
    #print(len(wordings))

def freq(sentences):
    freqs = {}
    for word in sentences:
        #for word in sentence:
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1
        #print(sentences)
        #print(max(freqs, key=freqs.get))
    return freqs


def plot_zipf(my_dict):
    y_val = [x[1] for x in my_dict]
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Zipf Curve")
    plt.ylabel("Frequency")
    plt.xlabel("Word index")
    plt.plot(y_val)
    plt.show()

#in texts[list], 0-3 are CEO, 4-7 are Current students,
# 8-11 are Juniors level, 12-15 are Recent Graduates, 16-19 are Senior level

#change the number in the square bracket to analyze
reuters_freq=freq(wordlist[19])


#print(wordlist[2])

reuters_dict=sorted(reuters_freq.items(),key=lambda x: x[1],reverse=True)
print(reuters_dict)

#plot_zipf(reuters_dict)




#NAMED RECOGNITION and POS
def ner(pdfwords):
    return (ne_chunk(pos_tag(word_tokenize(pdfwords))))

#in texts[list], 0-3 are CEO, 4-7 are Current students,
# 8-11 are Juniors level, 12-15 are Recent Graduates, 16-19 are Senior level

#change the number in the square bracket to analyze

print(ner(texts[16]))
#print(ner(texts[17]))
#print(ner(texts[18]))
#print(ner(texts[19]))


#SENTIMENT ANALYSIS

def sent(pdf):
    linkedin = TextBlob(pdf)
    return linkedin.sentiment.polarity

sentiment=[]
for i in range(0, len(texts)):
    pdf=texts[i]
    sentiment.append(sent(pdf))
    i+=1
print(sentiment)
#in texts[list], 0-3 are CEO, 4-7 are Current students,
# 8-11 are Juniors level, 12-15 are Recent Graduates, 16-19 are Senior level
#So I would average the numbers to find their average sentiment upto 4 decimals.


#TRIAL
"""

def pos(pdf):
    size = int(len(pdf) * 0.9)
    train_sents = pdf[:size]
    test_sents = pdf[size:]
    def_tagger = pdf.DefaultTagger("NN")
    uni_tagger = pdf.UnigramTagger(train_sents, backoff=def_tagger)
    print("\n")
    print("Results on train set {0}".format(uni_tagger.evaluate(train_sents)))
    print("Results on test set {0}".format(uni_tagger.evaluate(test_sents)))


# in texts[list], 0-3 are CEO, 4-7 are Current students,
# 8-11 are Juniors level, 12-15 are Recent Graduates, 16-19 are Senior level

# change the number in the square bracket to analyze
print(pos(texts))

"""