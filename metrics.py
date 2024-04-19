from statistics import mean
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score

class BLEUScore:
    """
    Classe para calcular o BLEU Score para avaliação de traduções automáticas.
    """

    def __init__(self, ngrams: int = 4) -> None:
        """
        Inicializa a classe com a quantidade de n-gramas e configura a suavização.
        :param ngrams: Número de n-gramas para calcular os pesos.
        """
        self.smoothing_function = SmoothingFunction().method3
        self.ngram_count = ngrams
        weights = [1 / ngrams if i <= ngrams else 0 for i in range(1, 5)]
        self.weights = tuple(weights)

    def __call__(self, references, hypothesis) -> float:
        """
        Calcula o BLEU score dada uma hipótese e suas referências.
        :param references: Listas de listas de tokens de referência.
        :param hypothesis: Lista de tokens da hipótese.
        :return: BLEU score como um float.
        """
        score = sentence_bleu(references, hypothesis, weights=self.weights, smoothing_function=self.smoothing_function)
        return score

    def __repr__(self) -> str:
        return f"bleu{self.ngram_count}"

class GLEUScore:
    """
    Classe para calcular o GLEU Score, uma variação do BLEU Score.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return sentence_gleu(*args, **kwargs)

    def __repr__(self):
        return "gleu"

class METEORScore:
    """
    Classe para calcular o METEOR Score, uma métrica alternativa ao BLEU.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return meteor_score(*args, **kwargs)

    def __repr__(self):
        return "meteor"

class Metrics:
    """
    Classe para gerenciar múltiplas métricas de avaliação de traduções.
    """

    def __init__(self) -> None:
        """
        Inicializa métricas BLEU com diferentes n-gramas, GLEU e METEOR.
        """
        bleu1 = BLEUScore(ngrams=1)
        bleu2 = BLEUScore(ngrams=2)
        bleu3 = BLEUScore(ngrams=3)
        self.bleu4 = BLEUScore(ngrams=4)

        self.gleu = GLEUScore()
        meteor = METEORScore()  # Necessário nltk.download('omw-1.4')

        self.all_metrics = [bleu1, bleu2, bleu3, self.bleu4, self.gleu, meteor]

    def calculate(self, refs: List[List[List[str]]], hypos: List[List[str]], train: bool = False) -> Dict[str, float]:
        """
        Calcula as métricas para um conjunto de referências e hipóteses.
        :param refs: Lista de referências para cada hipótese.
        :param hypos: Lista de hipóteses.
        :param train: Se verdadeiro, calcula apenas BLEU4 e GLEU.
        :return: Dicionário com os nomes das métricas e seus respectivos scores.
        """
        if train:
            score_functions = [self.bleu4, self.gleu]
        else:
            score_functions = self.all_metrics

        scores = {}
        for function in score_functions:
            scores[repr(function)] = mean([function(ref, hypo) for ref, hypo in zip(refs, hypos)])

        return scores
