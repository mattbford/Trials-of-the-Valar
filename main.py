from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.core.audio import SoundLoader
from kivy.graphics import Rectangle
import numpy as np
import pandas as pd
import re

# Predicts classification based on prior data
def ApplyMultinomialNB(V,prior,condprob,d):
    W = []
    score = [0,0,0,0]
    for x in d.split():
        if x in V:
            W.append(x)
    for i in range(0,4):
        score[i] = np.log(prior[i])
        for t in W:
            score[i] += np.log(condprob[i][t])
    result = max([score[0], score[1], score[2], score[3]])
    if result == score[0]:
        return 0, score[0]
    elif result == score[1]:
        return 1, score[1]
    elif result == score[2]:
        return 2, score[2]
    elif result == score[3]:
        return 3, score[3]
    return

# builds conditional probability matrix
def TrainMultiNomialNB(C, D):
    V = ExtractVocab(D)
    N = len(D)
    probability = [0,0,0,0]
    textc = [[],[],[],[]]
    condprob = [{},{},{},{}]
    for c in range(0, 4): # only has classes 0 and 1 for our purposes
        count = CountDocsInClass(C, c)
        probability[c] = count/N
        textc[c] = ConcatenateTextofAllDocsInClass(D, C, c)
        tct = {}
        for x in V:
            tct[x] = textc[c].count(x)
        for x in V:
            condprob[c][x] = (tct[x] + 1) / (len(textc[c]) + len(V))
    
    return V, probability, condprob

# gets all words in a class, returns list
def ConcatenateTextofAllDocsInClass(D, C, c):
    textInClass = []
    for i in range(0, len(D)):
        if str(c) == C[i]:
            for word in D[i].split():
                textInClass.append(word)
    return textInClass

# gets all unique words in data set
def ExtractVocab(D):
    words  = [] # unique words in sentences
    for line in D:
        for x in line.split():
            if x not in words:
                words.append(x)
    return words            

# counts number of instances of each class
def CountDocsInClass(C, c):
    count = 0
    for line in C:
        if str(c) in line:
            count += 1
    return count

# globals for training on build
D = "global"
C = "global"
V = "global"
probability = "global"
condprob = "global"

Builder.load_file('MainApp.kv')

# Play background music
sound = SoundLoader.load('music.mp3')
if sound:
    sound.play()
    sound.volume = 25
    sound.loop = True

def on_text(instance, value):
    print('The widget', instance, 'have:', value)

class mainScreen(Screen):
    pass

class freeScreen(Screen):
    pass

class minorScreen(Screen):
    pass

class majorScreen(Screen):
    pass

class maliceScreen(Screen):
    pass

# Create the screen manager
sm = ScreenManager()
sm.add_widget(mainScreen(name='main'))
sm.add_widget(freeScreen(name='free'))
sm.add_widget(minorScreen(name='minor'))
sm.add_widget(majorScreen(name='major'))
sm.add_widget(maliceScreen(name='malice'))

class MainApp(App):

    def processResults(self, text):
        D = re.sub('[.,/\\"\':;\[\]()\-\!\?\n]', '', text.rstrip().lower())
        guess, certainty = ApplyMultinomialNB(V, probability, condprob, D)

        print(guess)
        print(certainty)
        
        if guess == 0:
            sm.current = 'free'
        elif guess == 1:
            sm.current = 'minor'
        elif guess == 2:
            sm.current = 'major'
        else:
            sm.current = 'malice'

    def toggleAudio(self, slider):
        if slider.value > 1:
            slider.value = 0
        elif slider.value == 0:
            slider.value = 25

    def audioVolume(self, value):
        sound.volume = value/100
        self.value = value

    def build(self):
        return sm


# Create the textinput
textinput = TextInput()
textinput.bind(text=on_text)


if __name__ == "__main__":
    # train naiveBayes
    trainData = open("traindata.txt", "r")
    trainLabels = open("trainlabels.txt", "r")
    D = [re.sub('[.,/\\"\':;\[\]()\-\!\?\n]', '', line.rstrip().lower()) for line in trainData]
    C = [re.sub('[.,/\\"\':;\[\]()\-\!\?\n]', '', line.rstrip().lower()) for line in trainLabels]
    V,probability,condprob = TrainMultiNomialNB(C,D)
    trainData.close()
    trainLabels.close()

    MainApp().run()