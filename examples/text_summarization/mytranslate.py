from abc import ABC, abstractmethod 
from googletrans import Translator as GoogleEngine
from translate import Translator as MyMemEngine
import boto3
import json



class Translator:
    known_to_en_translations = {}
    known_to_de_translations = {}

    def __init__(self,path='') -> None:
        self.read_to_en_dict(path+"dict_en.json")
        #self.read_to_de_dict("dict_de.json")
        self.setup_engine()

    @abstractmethod
    def setup_engine(self) -> None:
        self.read_to_en_dict("dict_en.json")
        #self.read_to_de_dict("dict_de.json")
        self.setup_engine()

    @abstractmethod
    def request_to_english(self,text:str)->str:
        pass

    @abstractmethod
    def request_to_german(self,text:str)->str:
        pass

    def to_english(self,text:str) -> str:
        if text not in self.known_to_en_translations:
            self.known_to_en_translations[text] = self.request_to_english(text)
            self.save_to_en_dict("dict_en.json")
        return self.known_to_en_translations[text]
    
    def to_german(self,text:str) -> str:
        if text not in self.known_to_de_translations:
            self.known_to_de_translations[text] = self.request_to_german(text)
            #self.save_to_de_dict("dict_de.json")
        return self.known_to_de_translations[text]

    def save_to_en_dict(self, path:str):
        json.dump( self.known_to_en_translations, open( path, 'w' ) )

    def save_to_de_dict(self, path:str):
        json.dump( self.known_to_de_translations, open( path, 'w' ) )

    def read_to_en_dict(self, path:str):
        self.known_to_en_translations = json.load( open( path ) )

    def read_to_de_dict(self, path:str):
        self.known_to_de_translations = json.load( open( path ) )

class GoogleTranslator(Translator):
    engine_google = None

    def setup_engine(self) -> None:
        self.engine_google = GoogleEngine()

    def request_to_english(self,text:str) -> str:
        return self.engine_google.translate(text=text, src='de', dest='en').text
    def request_to_german(self,text:str) -> str:
        return self.engine_google.translate(text=text, src='en', dest='de').text

class AWSTranslator(Translator):
    def setup_engine(self) -> None:
        self.engine_aws = boto3.client('translate')

    def request_to_english(self,text:str) -> str:
        print("Sending a AWS request over "+str(len(text))+" characters.")
        sentences = text.split(".")
        batch = ""
        result = ""
        for sentence in sentences:
            if(len(batch)+len(sentence)>1000):
                result += self.engine_aws.translate_text(Text=batch,SourceLanguageCode='de',TargetLanguageCode='en')['TranslatedText']
                batch = sentence+"."
            else:
                batch += sentence+"."
        result += self.engine_aws.translate_text(Text=batch,SourceLanguageCode='de',TargetLanguageCode='en')['TranslatedText']
        return result

    def request_to_german(self,text:str) -> str:
        sentences = text.split(".")
        batch = ""
        result = ""
        for sentence in sentences:
            if(len(batch)+len(sentence)>500):
                result += self.engine_aws.translate_text(Text=batch,SourceLanguageCode='en',TargetLanguageCode='de')['TranslatedText']
                batch = sentence+"."
            else:
                batch += sentence+"."
        result += self.engine_aws.translate_text(Text=batch,SourceLanguageCode='en',TargetLanguageCode='de')['TranslatedText']
        return result

class MyMemoryTranslator(Translator):
    # they only accept up to 500 chars so we need to create batches of sentences.
    engine_english = None
    engine_german = None

    def setup_engine(self) -> None:
        self.engine_english = MyMemEngine(to_lang="en",from_lang="de")
        self.engine_german = MyMemEngine(to_lang="de",from_lang="en")

    def request_to_english(self,text:str) -> str:
        sentences = text.split(".")
        batch = ""
        result = ""
        for sentence in sentences:
            if(len(batch)+len(sentence)>500):
                result += self.engine_english.translate(text=batch)
                batch = sentence+"."
            else:
                batch += sentence+"."
        result += self.engine_english.translate(text=batch)
        return result

    def request_to_german(self,text:str) -> str:
        sentences = text.split(".")
        batch = ""
        result = ""
        for sentence in sentences:
            if(len(batch)+len(sentence)>500):
                result += self.engine_german.translate(text=batch)
                batch = sentence+"."
            else:
                batch += sentence+"."
        result += self.engine_german.translate(text=batch)
        return result