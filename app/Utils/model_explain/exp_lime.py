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
        self.indx_pred = np.argmax(self.exp.predict_proba)
        prediction = self.class_names[self.indx_pred]
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


    def get_word_pos_score(self):
        output = {}
        words_list = self.exp.as_list(self.indx_pred)
        words_map =  self.exp.as_map()[self.indx_pred]
        words_with_score = {}
        for i in range(len(words_list)):
            word_pos = words_map[i][0]
            word_name = words_list[i][0]
            word_score = words_map[i][1]
            words_with_score[word_name] = {'position':word_pos,'score':word_score}
        output['explanations'] = words_with_score
        return output

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




    