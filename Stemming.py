import nltk

# nltk.download("punkt")
from nltk.stem import PorterStemmer

# nltk.download("stopwords")
from nltk.corpus import stopwords

paragraph = "Hello. Truth is, I don't know what to say here. These recent days I've been thinking, giving lots of thought to what I could say. And the truth is that I can't think of anything. This is really difficult for me after so many years, being here my entire life. I'm not ready for this. And honestly, last year, with all the nonsense with the burofax and everything, I was convinced I knew what I wanted to say, but this year, this year is not the same. This year, my family and I were convinced that we were going to stay here, that we were going to stay at home. That's what we all wanted more than anything. We'd always made this our own. We were at home. We thought we'd be staying here in Barcelona. The time we've had here in the city and in the sport has been amazing. But well, today is I have to say goodbye to all of this. I've been here so many years, my entire life here, since I was 13. After 21 years, I'm leaving with my wife, with my three little Catalan-Argentine kids. And I can't tell you everything that we've lived in this city, and I can't say that in a few years, we won't come back because this is our home, and I promised my children that. I'm just really grateful for everything, all my teammates, all my former teammates, everything at the club, everyone that's been by my side, all the people that, there's so many people, even though some of us, I only met a few times. And yeah, this club I will always be so humble and have so much respect. I want to say to everyone at this house, for the luck that I've had to live so many experiences here at this club, so many beautiful things have happened, also some bad things, but all of this helped me to grow, help me to improve, and make me the person that I am today."

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [
        stemmer.stem(word)
        for word in words
        if word not in set(stopwords.words("english"))
    ]
    sentences[i] = " ".join(words)

print(sentences)
