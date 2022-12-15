"""This module for model explainabilty using lime"""

from lime.lime_text import LimeTextExplainer

class explainer():
    def __init__(self,model_predictor):
        """Use this class for explaining predictions
        Args:
        model_predictor: is the model predictor class
        """
        self.model_predictor=model_predictor
        self.class_names = self.model_predictor.get_class_names()
        print("class names are: ",self.class_names)
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_texts(self,text,top_labels=None):
        num_feature=50
        exp = self.explainer.explain_instance(text,
                    self.model_predictor.predict_explain,
                    num_features=num_feature,
                    top_labels=top_labels)
        return exp