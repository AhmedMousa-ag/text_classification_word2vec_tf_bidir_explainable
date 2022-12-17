"""This module for model explainable using lime"""

from lime.lime_text import LimeTextExplainer
import numpy as np
import json
import os
import glob

# TODO document module file

class explainer():
    def __init__(self, model_predictor):
        """Use this class for explaining predictions
        Args:
        model_predictor: is the model predictor class
        """
        self.model_predictor = model_predictor
        self.class_names = self.model_predictor.get_class_names()
        print("class names are: ", self.class_names)
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_texts(self, text: str, top_labels=None):
        num_feature = 20
        self.exp = self.explainer.explain_instance(text,
                                                   self.model_predictor.predict_explain,
                                                   labels=range(
                                                       len(self.class_names)),
                                                   num_features=num_feature,
                                                   top_labels=top_labels)
        return self.exp

    def get_prediction(self):
        self.indx_pred = np.argmax(self.exp.predict_proba)
        prediction = self.class_names[self.indx_pred]
        return prediction

    def get_labels_score(self):
        output = {}
        labels_with_score = {}
        predic_proba = self.exp.predict_proba
        for indx, label in enumerate(self.class_names):
            labels_with_score[label] = str(predic_proba[indx])
        output["scores"] = labels_with_score
        return output

    def get_word_pos_score(self):
        words_list = self.exp.as_list(self.indx_pred)
        words_map = self.exp.as_map()[self.indx_pred]
        words_with_score = {}
        for i in range(len(words_list)):
            word_pos = str(words_map[i][0])
            word_name = str(words_list[i][0])
            word_score = str(words_map[i][1])
            words_with_score[word_name] = {
                'position': word_pos, 'score': word_score}

        return words_with_score

    def produce_explainations(self, data):
        output = {}
        id_col, text_col, targ_col = get_id_text_targ_col()
        ids = data[id_col]
        texts = data[text_col]
        pred_list = []
        for id, txt in zip(ids,texts):
            result = {}
            print(f"raw text: {txt}")
            self.explain_texts(text=txt)
            result[id_col] = id
            result[targ_col] = self.get_prediction()
            result['scores'] = self.get_labels_score()
            result['explanations'] = self.get_word_pos_score()
            pred_list.append(result)

        output['predictions'] = pred_list
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


def read_data_config_schema():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    #path = glob.glob(os.path.join(os.pardir, "ml_vol", 'inputs', 'data_config', '*.json'))[0] #TODO uncomment
    path = glob.glob(os.path.join("opt", "ml_vol", 'inputs', 'data_config', '*.json'))[0] 
    try: 
        json_data = json.load(open(path))
        return json_data
    except:
        raise Exception(f"Error reading json file at: {path}")

def get_id_text_targ_col():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    schema = read_data_config_schema()
    id_col = schema['inputDatasets']['textClassificationBaseMainInput']['idField']
    text_col = schema['inputDatasets']['textClassificationBaseMainInput']['documentField']
    targ_col = schema['inputDatasets']['textClassificationBaseMainInput']['targetField']
    return id_col, text_col, targ_col
