# Machine Learning Project: Calibration of Probabilities

## Project proposal: Assignment 2 
The course develops supervised learning primarily through the lens of risk minimization, where models are trained to minimize a surrogate loss such as the logistic or hinge loss. In this framework, the output of a classifier is often interpreted as a score or a probability, and performance is typically evaluated in terms of classification accuracy or expected loss.

However, in many applications it is not enough to predict the correct label; one also needs reliable probability estimates. A model is said to be calibrated if, among all predictions made with confidence 0.8, roughly 80% are correct. Standard learning algorithms, even when optimized for log-loss, can produce poorly calibrated probabilities, especially when they are overconfident or when the model class is misspecified. For more information, refer to this blog post.

The paper by Niculescu-Mizil and Caruana addresses this gap by systematically studying calibration and proposing simple post-processing methods to improve it. This extends the course material by shifting the focus from prediction accuracy to the quality of uncertainty estimates, and by showing how risk minimization alone does not guarantee well-calibrated outputs. It provides a concrete bridge between theoretical loss functions and practical evaluation metrics.

Apprendimento supervisionato per minimizzare il rischio --> l'output della classificazione è una percentuale (valutate in termini di accuratezza di classificazione)
Servono stime di probabilità affidabili --> modello calibrato se le previsioni risultano corrette all'80%
la sola minimizzazione del rischio non garantisca risultati ben calibrati

## Objective
Evaluate and improve the calibration of probabilistic classifiers

Questo assignment non vuole solo vedere se un modello classifica bene, ma se le sue probabilità sono affidabili.

Se il modello dice “questa osservazione è positiva con probabilità 80%”, allora, tra tutti i casi in cui dice 80%, dovrebbe avere ragione circa l’80% delle volte. Questa è l’idea di calibration: le probabilità stimate devono corrispondere alla realtà osservata.

Bisogna mostrare che un modello può avere una buona accuracy ma dare probabilità sbagliate.
Non basta predire la classe giusta: bisogna anche capire se il modello è affidabile quando dice “sono sicuro al 70%, 80%, 90%.

## Dataset
Two real-world classification datasets:

- load_breast_cancer in scikit-learn --> più semplice quindi puoi iniziare da questo che sarà più calibrato
  (from sklearn.datasets import load_breast_cancer
  cancer = load_breast_cancer() )
- fetch_openml in scikit-learn --> più complicato quindi calibrazione interessante
  (from sklearn.datasets import fetch_openml
  diabetes = fetch_openml(name='diabetes', version=1, as_frame=True) )

## Tasks

### Train: --> allenare 2 modelli
- Logistic Regression --> modello base più interpretabile e lineare
- Random Forest --> modello più flessibile, composto da tanti alberi decisionali

Conviene dividere i dati in tre parti:
- Train set: Serve per allenare il modello iniziale → alleni Logistic Regression e Random Forest
- Calibration set: Serve per imparare la correzione delle probabilità → applichi Platt scaling e isotonic regression
- Test set: Serve solo alla fine, per valutare tutto in modo corretto → confronti risultati finali

Questo è importante perché il paper dice che usare lo stesso dataset sia per allenare il modello sia per calibrarlo può introdurre bias; per questo serve un set indipendente di calibrazione --> Calibrare significa prendere le probabilità grezze del modello e modificarle per renderle più realistiche
Quindi la calibrazione non cambia necessariamente la classe predetta, ma corregge la fiducia del modello

### Apply: --> metodi di calibrazione
- Platt scaling: Prende gli output del modello e li passa dentro una funzione sigmoidale --> più stabile con pochi dati e meno flessibile --> meglio se ho pochi dati
- Isotonic regression: Cerca solo una trasformazione crescente: se un caso aveva probabilità più alta di un altro prima, deve rimanere più alta anche dopo --> più flessibile e corregge distorsioni più complesse ma ha più rischio di overfitting se il calibration set è piccolo --> meglio se ho tanti dati

### Evaluate: --> confrontare i modelli prima e dopo la calibrazione usando tre metriche
- Accuracy: Misura quante classi vengono predette correttamente ma l’accuracy non dice se le probabilità sono buone
- Log-loss: Misura quanto sono buone le probabilità
- Brier score: misura la distanza tra probabilità prevista e risultato reale

### Plot:
- Reliability diagrams: Serve a vedere visivamente se il modello è calibrato, Se il modello è ben calibrato, i punti stanno vicino alla diagonale

## Expected Output
- Calibration curves before and after correction
- Quantitative comparison of metrics
- Discussion of miscalibration

Quindi:
- prima della calibrazione, allenare i modelli e valutare
- applicare platt ai modelli e valutare
- applicare isotonic ai modelli e valutare
- fai tabelle di comparazione

I risultati mostrano che valutare solo l’accuracy non è sufficiente. Due modelli possono classificare in modo simile, ma avere probabilità molto diverse in termini di affidabilità. La calibrazione permette di correggere queste probabilità e rende il modello più utile in contesti in cui la probabilità prevista viene usata per prendere decisioni.

Struttura report:
1.	Introduzione: spieghi cos’è la calibrazione; spieghi perché non basta l’accuracy.
2.	Riferimento teorico: paper di Niculescu-Mizil e Caruana; Platt Scaling; Isotonic Regression; reliability diagrams.
3.	Dataset: descrizione dei due dataset; target; numero di osservazioni; preprocessing.
4.	Metodologia: train/calibration/test split; Logistic Regression; Random Forest; metriche usate.
5.	Risultati: tabelle metriche; reliability diagrams prima/dopo; confronto tra Platt e Isotonic.
6.	Discussione: quale modello era più calibrato; quale calibrazione ha funzionato meglio; differenza tra accuracy e qualità delle probabilità.
7.	Conclusione: calibrazione utile; non sempre migliora tutto; va valutata con metriche adatte.

Ordine pratico con cui iniziare
1.	carica un solo dataset;
2.	fai train/calibration/test split;
3.	implementa Logistic Regression da zero;
4.	calcola accuracy, log-loss, Brier score;
5.	fai reliability diagram;
6.	implementa Platt Scaling;
7.	implementa Isotonic Regression;
8.	solo dopo ripeti tutto sul secondo dataset;
9.	solo alla fine aggiungi Random Forest.

Calibration_of_Probability/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── breast_cancer_analysis_01.ipynb
│   └── diabetes_analysis_02.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── calibration.py
│   └── visualization.py
│
└── presentation/
    └── probability_calibration_report.pdf
