import pandas
from sklearn.model_selection import train_test_split
from bert_score import BERTScorer
from rouge import Rouge 
from typing import Callable
import json
import sys
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from statistics import mean 
from torch import Tensor

class Workbench:
    test_source = []
    test_summary = []
    scores = []
    results = {}

    def __init__(self) -> None:
        pd = pandas.read_csv("/home/marc/Nextcloud/Uni/20SoSe/DI-Lab/Data/swisstext/data_train.csv")
        train = pd.values.tolist()
        source = [item[0] for item in train]
        summary = [item[1] for item in train]
        self.train_source,self.test_source,self.train_summary,self.test_summary=train_test_split(source,summary,train_size=0.95,test_size=0.005,random_state=123)
        self.reset()
        self.no_bertscore()
        self.add_bertscore()
 

    def no_bertscore(self):
        rouge_scorer = Rouge()
        def r1_score(hypothesis:str,reference:str):
            scores_rouge = rouge_scorer.get_scores(hypothesis, reference)[0]
            return scores_rouge["rouge-1"]["f"]

        def r2_score(hypothesis:str,reference:str):
            scores_rouge = rouge_scorer.get_scores(hypothesis, reference)[0]
            return scores_rouge["rouge-2"]["f"]

        def rl_score(hypothesis:str,reference:str):
            scores_rouge = rouge_scorer.get_scores(hypothesis, reference)[0]
            return scores_rouge["rouge-l"]["f"]

        self.scores = [r1_score,r2_score,rl_score]

    def add_bertscore(self):
        bert_scorer = BERTScorer(num_layers=12,model_type="bert-base-german-cased")
        def bert_score(hypothesis:str,reference:str):
            P, R, F1 = bert_scorer.score([hypothesis],[reference])
            return Tensor.item(F1)
        self.scores.append(bert_score)

    def benchmark(self,model:Callable[[str], str],num_samples=-1):
        self.populate_summaries(model,num_samples)
        self.populate_scores(model.__name__)

    def populate_summaries(self,model:Callable[[str], str],num_samples:int):
        if(num_samples==-1):
            num_samples = len(self.test_source)

        self.results["summaries"][model.__name__] = []
        for index in range(0,num_samples):
            sys.stdout.flush()
            sys.stdout.write('\r')
            percent = int((index+1)/num_samples*100)
            sys.stdout.write(("[%-20s] %d%%"+" ("+str(index+1)+"/"+str(num_samples)+") summarizing with "+model.__name__) % ('='*int(percent/5), percent))
            hypothesis = model(self.test_source[index])
            self.results["summaries"][model.__name__].append(hypothesis)

    def populate_scores(self,model_name):
        #initialize results for this model
        for score in self.scores:
            if score.__name__ not in self.results.keys():
                self.results[score.__name__] = {}
            self.results[score.__name__][model_name] = []

        num_samples = len(self.results["summaries"][model_name])
        for index in range(0,num_samples):
            hypothesis = self.results["summaries"][model_name][index]
            reference = self.test_summary[index]
            for score in self.scores:
                sys.stdout.flush()
                sys.stdout.write('\r')
                percent = int((index+1)/num_samples*100)
                sys.stdout.write(("[%-20s] %d%%"+" ("+str(index+1)+"/"+str(num_samples)+") evaluating "+score.__name__+" (on "+model_name+")") % ('='*int(percent/5), percent))
                self.results[score.__name__][model_name].append(score(hypothesis,reference))

    def test_sample(self,model:Callable[[str],str]):
        text = "Herr  Präsident!  Liebe  Kolleginnen  und  Kollegen!  In dieser Krise zeigt sich eine gewisse Doppelgesichtigkeit: Auf  der  einen  Seite  beweisen  der  Sozialstaat  und  die Sozialversicherungen in diesen Monaten, dass sie selbst in  der  jetzigen  Extremsituation  mit  Milliardeneinsatz  in der Lage sind, in der Krise ein gewisses Maß an Sicherheit zu geben. Auf der anderen Seite zeigt das umwälzende  Ausmaß  der  Krise  auch,  dass  all  diese  Gegenmaßnahmen   die   Erschütterungen   durch   Arbeitslosigkeit, Existenzverlust  und  Bildungsmangel  nicht  vollständig auffangen können. Tiefe Gräben tun sich auf. Es sind gerade die Verwundbarsten und die ärmsten Gruppen, die ohnehin am Rande der  Gesellschaft  stehen,  die  nun  umso  mehr  den  Anschluss verlieren: Menschen, bei denen das Kurzarbeitergeld nicht reicht, Familien, die Covid-19 vor fast unlösbare  Probleme  stellt,  Frauen,  die  den  größten  Teil  der Sorgearbeit leisten müssen, Wohnungslose ohne jede Unterstützung oder Studierende und Auszubildende, denen ein Abbruch der Ausbildung droht. Die Pandemie hinterlässt  bei  vielen  Menschen  ein  Gefühl  von  Einsamkeit, von persönlicher Überforderung und sogar Verzweiflung. Damit ist die Coronakrise auch eine Krise des gesellschaftlichen  und  sozialen  Zusammenhalts.  Darauf  müssen wir reagieren, und zwar mit einem Sozialschutz-Paket III, das diese Lücken schließt und das aber auch und vor allem über die Krise hinausweist. Wir brauchen einen Dreiklang der Garantien. Dazu  gehört  erstens  eine  materielle  Basissicherung. Die  heutige  Grundsicherung  muss  zu  einer  Garantiesicherung  mit  höheren  Regelsätzen  und  ohne  Sanktionen weiterentwickelt werden. Zweitens  müssen  wir  die  Sozialversicherungen  stärken,  unter  anderem  mit  einem  Rettungsschirm  für  die Sozialversicherungen, damit die Einnahmeausfälle durch die Coronakrise nicht zu Leistungskürzungen führen. Drittens wird in dieser Krise ganz deutlich: Materielle Unterstützung allein reicht nicht. Gerade in Krisenzeiten erfordert  gesellschaftliche  Teilhabe  mehr  als  finanzielle Absicherung. Notwendig ist eine Strategie, die jede und jeden dabei unterstützt, mit kommenden Unsicherheiten, die durch die ganzen anderen Krisen, die wir ja auch noch haben,  ebenfalls  verursacht  werden,  besser  umgehen  zu können – eine Strategie der Förderung von Widerstandsfähigkeit, eine Strategie der Förderung von Resilienz. Dafür  braucht  es  Orte  der  Begegnung,  Orte  der  Gemeinschaft, an denen Menschen Unterstützung und Anerkennung erfahren. Dienste und Initiativen der sozialen Arbeit,  der  Weiterbildung  und  der  personenbezogenen Unterstützung sind dazu unverzichtbar. Diese Förderung der Erfahrung von Selbstwirksamkeit hilft Menschen auf die Beine. Diese Stärkung der eigenen Ressourcen kann in  großen  Institutionen  stattfinden  wie  in  der  von  uns vorgeschlagenen  Arbeitsversicherung.  Sie  spielt  sich aber  auch  genauso  an  vielen  kleinen  Orten  ab.  Solche Orte  leisten  das,  was  eine  Überweisung  auf  ein  Konto nicht  kann:  Sie  stärken  die  Menschen  am  Rand.  Diese Orte stabilisieren gleichzeitig aber auch die gesellschaftliche Mitte und tragen damit zu einer gemeinwohlorientierten Politik bei, die die Demokratie stärkt. Abschließend will ich noch ein Beispiel für einen solchen Ort geben, damit wir uns das besser vorstellen können:  In  meinem  Wahlkreis  gibt  es  das  Mütterzentrum Dortmund  als  Verein.  Dort  begegnen  sich  Mütter  mit Kindern,  Migranten,  Arbeitslose,  die  dort  zum  Beispiel in  öffentlich  geförderter  Beschäftigung  arbeiten.  Dort gibt  es  Angebote  für  Familien,  Angebote  der  Bildung, der  Beratung,  der  Schwangerenberatung,  eine  Musikschule und ein Café. Ich habe die Geschäftsführerin gefragt,  was  sie  eigentlich  braucht,  um  diesen  lebendigen Ort  des  Zusammenhalts  zu  stärken,  und  sie  sagte  mir: Solch einen Ort brauchen wir eigentlich in jedem Stadtviertel.  –  Das  hat  mich  tief  beeindruckt.  Ich  finde,  wir brauchen  ein  Sozialschutz-Paket  III,  das  diesen  Dreiklang der Garantien beinhaltet und diese Orte stärkt. Vielen Dank."
        return model(text)

    def reset(self):
        self.results = {"summaries":{}}

    def save(self,path:str):
        json.dump( self.results, open( path, 'w' ) )
        self.reset()

    def load(self,path:str):
        other = json.load( open( path ) )
        for score in other:
            if score not in self.results:
                self.results[score]={}
            for model in other[score]:
                if model not in self.results[score]:
                    print("Loaded model "+model+" with metric "+score+".")
                    self.results[score][model] = other[score][model]

    def plot(self):
        scores_to_plot = ["bert_score","r1_score","rl_score"]
        scores_to_plot = [score for score in scores_to_plot if score in self.results]
        fig = make_subplots(rows=len(scores_to_plot), cols=1,subplot_titles=[score for score in scores_to_plot])
        index = 0
        for score in scores_to_plot:
            if score not in scores_to_plot:
                continue
            index = index +1
            for model in self.results[score]:
                fig.add_trace(go.Histogram(x=self.results[score][model],name=model+ " (" + str(round(mean(self.results[score][model]),2)) +")", legendgroup = score, cumulative_enabled=False),row=index, col=1)
        fig.update_traces(opacity=0.5)
        fig.update_layout(barmode='overlay')
        fig.show()