"""This module for model explainabilty using lime"""

from lime.lime_text import LimeTextExplainer
import numpy as np
import tensorflow as tf

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
        #TODO handle Id
        num_feature=20
        self.exp = self.explainer.explain_instance(text,
                    self.model_predictor.predict_explain,
                    labels=range(len(self.class_names)),
                    num_features=num_feature,
                    top_labels=top_labels)
        return self.exp

    def get_prediction(self):
        #TODO catch exception if didn't call explain_texts() first
        self.index = np.argmax(self.exp.predict_proba)
        prediction = self.class_names[self.index]
        return prediction
    

    # TODO a function that gets all labels with their score
    def get_labels_score(self):
        output = {}
        
        labels_with_score = {}
        predic_proba = self.exp.predict_proba
        for indx,label in enumerate(self.class_names):
            labels_with_score[label] = predic_proba[indx]


        output["scores"] = labels_with_score
        return output
    # TODO a function that uses exp.get_list(self.index) to get names of the word

    # TODO a function that gets words place exp.as_map()[self.index]

    # TODO a function that combine them all together and return:
    '''{
    "predictions":[
        {
            "Id":3245,
            "label":"ham",
            "scores":{
                "ham": score,
                "spam":score
                },
            "explanations":{
                "token1":{
                    "position":"index",
                    "score":score
                },

                "token2":{
                    "position":"index",
                    "score":score
                }
            }
            }
        ]
    }
    '''




    