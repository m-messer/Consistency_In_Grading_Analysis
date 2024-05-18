import pandas as pd
from sentence_transformers import SentenceTransformer, util
from itertools import combinations


def calculate_sentence_sim(row):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embedding_1 = model.encode(row['feedback_1'], convert_to_tensor=True)
    embedding_2 = model.encode(row['feedback_2'], convert_to_tensor=True)

    return util.pytorch_cos_sim(embedding_1, embedding_2).detach().numpy()[0][0]


class SimilarityAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def preprocess(self):
        self.data = self.data.dropna()

    def save_data(self, output_path):
        self.data.to_csv(output_path, index=False)


class InterRaterSimilarityAnalysis(SimilarityAnalysis):

    def calculate_similarity(self):
        print("Preprocessing data...")
        self.preprocess()
        print("Formating table for intra_rater")
        # TODO
        print("Calculating pair-wise similarity...")
        # TODO
        print("Saving....")
        self.save_data('intra_rater_sim.csv')

    def generate_feedback_pairs(self, group, skill, assignment_number):
        pairs = combinations(range(int(group[0]), int(group[-1]) + 1), r=2)

        feedback_pairs_df = None

        for idx, pair in enumerate(pairs):
            if feedback_pairs_df is None:
                feedback_pairs_df = pd.DataFrame({'group': group,
                                                  'assignment_number': assignment_number,
                                                  'participant_id_1': pair[0],
                                                  'participant_id_2': pair[1],
                                                  'comment_1': self.data[(self.data['group'] == group) &
                                                                         (self.data['skill'] == skill) &
                                                                         (self.data[
                                                                              'assignment_number'] == assignment_number) &
                                                                         (self.data['participant_id'] == pair[0])]
                                                  ['comments'],
                                                  'comment_2': self.data[(self.data['group'] == group) &
                                                                         (self.data['skill'] == skill) &
                                                                         (self.data[
                                                                              'assignment_number'] == assignment_number) &
                                                                         (self.data['participant_id'] == pair[1])]
                                                  ['comments']
                                                  }, index=[idx])
            else:
                feedback_pairs_df = pd.concat([
                    feedback_pairs_df,
                    pd.DataFrame({'group': group, 'assignment_number': assignment_number,
                                  'participant_id_1': pair[0],
                                  'participant_id_2': pair[1],
                                  'comment_1': self.data[(self.data['group'] == group) &
                                                         (self.data['skill'] == skill) &
                                                         (self.data['assignment_number'] == assignment_number) &
                                                         (self.data['participant_id'] == pair[0])]
                                  ['comments'],
                                  'comment_2': self.data[(self.data['group'] == group) &
                                                         (self.data['skill'] == skill) &
                                                         (self.data['assignment_number'] == assignment_number) &
                                                         (self.data['participant_id'] == pair[1])]
                                  ['comments']
                                  }, index=[idx])]
                )

        return feedback_pairs_df.dropna()


class IntraRaterSimilarityAnalysis(SimilarityAnalysis):
    DUPLICATE_MAP = {
        680: 144,
        681: 559,
        682: 97,
        683: 358,
        684: 389,
        685: 160,
        686: 176
    }

    def calculate_similarity(self):
        print("Preprocessing data...")
        self.preprocess()
        print("Formating table for intra_rater")
        self.process_intra_rater()
        print("Calculating intra-rater similarity...")
        self.data['sim'] = self.data.apply(calculate_sentence_sim, axis=1)
        print("Saving....")
        self.save_data('intra_rater_sim.csv')

    def process_intra_rater(self):
        intra_rater_df = self.data[self.data['assignment_number'].isin(
            list(self.DUPLICATE_MAP.keys()) +
            list(self.DUPLICATE_MAP.values()))][['assignment_number', 'skill', 'participant_id', 'batch', 'comments']]

        intra_rater_df['assignment_number'] = (
            intra_rater_df['assignment_number'].apply(
                lambda x: x if x not in self.DUPLICATE_MAP.keys() else self.DUPLICATE_MAP[x]))

        pivot = intra_rater_df.pivot(index=['assignment_number', 'skill', 'participant_id'],
                                     columns='batch', values='comments').reset_index()
        pivot.columns = ['assignment_number', 'skill', 'participant_id', 'feedback_1', 'feedback_2']
        pivot = pivot.fillna('No comment supplied')

        self.data = pivot


if __name__ == '__main__':
    # TODO: Add args
    sa = InterRaterSimilarityAnalysis('data/intra_rater_data.csv')
    sa.calculate_similarity()
