# Recommender Systems with Alex, 22.04.2022


Kurzprotokol zum Meeting mit Alex betreffend recommender Systeme

---
## <span > __Basic Overview__ </span>
 

* Problembeschreibung
* Models
* Libraries 

---
## __Problembeschreibung__ 

Das Thema recommender System ist größer als gedacht und es gibt erheblich mehr Möglichkeiten, als wir überhaupt wissen. Da ist es gut an die Hand genommen zu werden und von jemanden guided zu werden, um nicht in die selben pitfalls zu fallen, wie die vor uns :) 

Als Literatur :

[Recommender systems handbook](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf)

[Recommender Systems:
An Introduction](http://pzs.dstu.dp.ua/DataMining/recom/bibl/1jannach_dietmar_zanker_markus_felfernig_alexander_friedrich.pdf)

[ Fashion Recommendation Systems, Models and Methods: A Review ](https://www.mdpi.com/2227-9709/8/3/49/htm)

[Fashion Recommendation System](https://thecleverprogrammer.com/2020/08/16/fashion-recommendation-system/)

## __Fragen__

* Wie müssen die Daten aufbereitet werden um in die Models gefüttert zu werden?
* Wie geht das System mit NaN um ?
    *könnten wir auf 0 setzen
    *mean normalisation ? (kein scaling) 
* Welche Modelle gibt es ? 
* Welche machen sinn?
* Was sollte man vermeiden?
* libaries ?
    * Surprise
    * regmetric
* Macht es Sinn mit Content-based-filtering oder Collaborative-Filtering zu starten? 
*Kombiniert man beides. Was macht bei unseren Daten am meisten Sinn?
* Wie wandelt man Käufe am besten in "Ratings" um?

* Fancy reinforced learning ? 

[* Top-K Off-Policy Correction for a REINFORCE Recommender System](https://towardsdatascience.com/top-k-off-policy-correction-for-a-reinforce-recommender-system-e34381dceef8)


## __Einleitung__

* Was haben wir? (Situation)
 Viele Daten (siehe: Columnsdescription)
 * Entscheidungen für Methoden ( Task)
    [Siehe](#models)
* Wie kommen wir dahin ?  (Action)
* Wo wollen wir hin?  (Result)



image recognition?
* ( side quest !)

### __Models__


[recommender-systems-in-practice (Overview article)](https://medium.com/towards-data-science/recommender-systems-in-practice-cef9033bb23a)

[recommender-systems](https://d2l.ai/chapter_recommender-systems/index.html)

[NVIDIA: How to Build a Winning Recommendation System, Part 1](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-1/)

[NVIDIA: How to Build a Deep Learning Powered Recommender System, Part 2](https://developer.nvidia.com/blog/how-to-build-a-winning-recommendation-system-part-2-deep-learning-for-recommender-systems/)

1. Content based
2.  Collerativ filtering
    * [cold start problem](https://www.researchgate.net/profile/Mehdi-Elahi-2/publication/343906250_Fashion_Recommender_Systems_in_Cold_Start/links/5f478d43299bf13c503d9b19/Fashion-Recommender-Systems-in-Cold-Start.pdf?_sg%5B0%5D=P7yn6VDQRboc0lGPqabSt3CRtnwFIQCK04d28P8UutS886g8QI8Xza5tbXn9DouDjjLn5y4TkoJ_zWM3Re3Kow.SQmAR5V48OK2icrhEqvNX4QgUWjcwPM9msKreApMgXWIFBaPfXxWviITMQ2snD4FYO_LCkJKVGNwcCOezT_EoQ&_sg%5B1%5D=yBwrQeH49sdTiyF55WIpQ8pk7cottXZAPnivOJ7kxllvsZ13EBuIKibPeMhTKkozvU25fyYWjykBSiomwD6smxCzuzXXsUNe907Bgcq4BI2K.SQmAR5V48OK2icrhEqvNX4QgUWjcwPM9msKreApMgXWIFBaPfXxWviITMQ2snD4FYO_LCkJKVGNwcCOezT_EoQ&_iepl=)
        * content based model
    * NLP (gibt content)
        *clustering

    KNN zuordnen, um die Zuordnung auf Basis von Alter, PLZ etc.?


[Fashion Item Classification and Recommendation System with NLP Techniques](https://medium.com/mlearning-ai/fashion-item-classification-and-recommendation-system-with-nlp-techniques-c1cfd4eecc98)


####implicit recommender systems:

[implicit (libary)](https://github.com/benfred/ç)

#### Baskezt size

[basket size solution](https://opensourceconnections.com/blog/2016/06/06/recommender-systems-101-basket-analysis/)

[Basket size with coding example ](https://pbpython.com/market-basket-analysis.html)

"Andere kunden kauften auch" Algorythmus
    * Clustern

#### Hit rate 

[Evaluating A Real-Life Recommender System, Error-Based and Ranking-Based](https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b)

[Beinhaltet N-Top](https://www.youtube.com/watch?v=EeXBdQYs0CQ)

[Recommendation Algorithms for Optimizing
Hit Rate, User Satisfaction and Website Revenue](https://www.ijcai.org/Proceedings/15/Papers/259.pdf)

### Matrix factorization (recommender systems)
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.[1] This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post,[2] where he shared his findings with the research community. The prediction results can be improved by assigning different regularization weights to the latent factors based on items' popularity and users' activeness. - (Source: [Wiki article (surprised)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

[Recommendation Systems: Collaborative Filtering using Matrix Factorization — Simplified (short, good _medium_ article)](https://medium.com/p/2118f4ef2cd3)

### Singular Value Decomposition (SVD)

Eine Singulärwertzerlegung (engl. Singular Value Decomposition; abgekürzt SWZ oder SVD) einer Matrix bezeichnet deren Darstellung als Produkt dreier spezieller Matrizen. Daraus kann man die Singulärwerte der Matrix ablesen. Diese charakterisieren, ähnlich den Eigenwerten, Eigenschaften der Matrix. Singulärwertzerlegungen existieren für jede Matrix – auch für nichtquadratische Matrizen. (Source: [https://de.wikipedia.org/wiki/Singul%C3%A4rwertzerlegung](https://de.wikipedia.org/wiki/Singul%C3%A4rwertzerlegung))

[Singular Value Decomposition (SVD) & Its Application In Recommender System](https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/)

[Using Singular Value Decomposition to Build a Recommender System](https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/)

### Splitting der Modelle ? 

[7 Types of Hybrid Recommendation System](https://medium.com/analytics-vidhya/7-types-of-hybrid-recommendation-system-3e4f78266ad8)

[A Guide to Building Hybrid Recommendation Systems for Beginners](https://analyticsindiamag.com/a-guide-to-building-hybrid-recommendation-systems-for-beginners/)

[Creating a Hybrid Content-Collaborative Movie Recommender Using Deep Learning](https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af)

### Deeplearn

[Deep Learning based Recommender Systems](https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e)

[Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf)


### Streamlit

[Streamlit page](https://streamlit.io/)


### cold start problem

Kein Item gegeben (für _9950_ Kunden, die nicht in der Trans_liste sind) Raus geschmissen.

---
## Libraries

[Surprise](http://surpriselib.com/)

Surprise or Simple Python RecommendatIon System Engine is a Python SciPy toolkit for building and analysing recommender systems. The tool deals with explicit rating data. With a set of built-in algorithms and datasets Surprise can help you learn how to build recommender systems. It provides various ready-to-use prediction algorithms such as baseline algorithms, neighbourhood methods, matrix factorisation-based such as SVD, PMF, SVD++, NMF and many others. The features of surprise include easy dataset handling, easy to implement new algorithm ideas, among others.

[Lenskit](https://lenskit.org/)

LensKit is an open-source toolkit for building, researching, and learning about recommender systems. It provides support for training, running, and evaluating recommender algorithms in a flexible fashion suitable for research and education. LensKit for Python (also known as LKPY) is the successor to the Java-based LensKit toolkit and a part of the LensKit project. It enables researchers to build robust and reproducible experiments that can make use of the growing PyData and Scientific Python ecosystem, including Scikit-learn, TensorFlow, and PyTorch. 

[Crab](http://muricoca.github.io/crab/)

rab as known as scikits.recommender is a Python framework for building recommender engines integrated with the world of scientific Python packages (numpy, scipy, matplotlib).

The engine aims to provide a rich set of components from which you can construct a customized recommender system from a set of algorithms and be usable in various contexts: ** science and engineering ** .

[Rexy](https://github.com/mazdaka/Rexy)
![](./images/Meeting_Alex_Rexy.jpg)

Rexy (rec-sy) is an open-source recommendation system based on a general User-Product-Tag concept and a flexible structure that has been designed to be adaptable with various data-schema. There are a lot of methods and ways that Rexy has used to recommend the products to users. This includes general recommendations like Top products, event based recommendations and Novel products that the user might be interested in. There are other recommendations that are directly related to the user's activities or other users that have a similar behavior to the given user.

[TensorRec](https://github.com/jfkirk/tensorrec)

A TensorFlow recommendation algorithm and framework in Python.

__NOTE: TensorRec is not under active development__
TensorRec will not be receiving any more planned updates. Please feel free to open pull requests -- I am happy to review them.

Er empfielt :

[TensorFlow Ranking](https://github.com/tensorflow/ranking/)

Spotlight 

und LightFM

[LightFM](https://github.com/lyst/lightfm)

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

[Case Recommender](https://github.com/caserec/CaseRecommender)

Case Recommender is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback. The framework aims to provide a rich set of components from which you can construct a customized recommender system from a set of algorithms. Case Recommender has different types of item recommendation and rating prediction approaches, and different metrics validation and evaluation.


[Spotlight](https://maciejkula.github.io/spotlight/)

Spotlight uses PyTorch to build both deep and shallow recommender models. By providing both a slew of building blocks for loss functions (various pointwise and pairwise ranking losses), representations (shallow factorization representations, deep sequence models), and utilities for fetching (or generating) recommendation datasets, it aims to be a tool for rapid exploration and prototyping of new recommender models.


---

