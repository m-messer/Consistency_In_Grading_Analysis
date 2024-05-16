import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SimilarityAnalysis:
    DUPLICATE_MAP = {
        680: 144,
        681: 559,
        682: 97,
        683: 358,
        684: 389,
        685: 160,
        686: 176
    }

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def calculate_intra_rater_similarity(self):
        print("Preprocessing data...")
        self.preprocess()
        print("Formating table for intra_rater")
        self.process_intra_rater()
        print("Calculating intra-rater similarity...")
        self.data['sim'] = self.data.apply(self.calculate_sentence_sim, axis=1)
        print("Saving....")
        self.save_data()

    def calculate_sentence_sim(self, row):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        embedding_1 = model.encode(row['feedback_1'], convert_to_tensor=True)
        embedding_2 = model.encode(row['feedback_2'], convert_to_tensor=True)

        return util.pytorch_cos_sim(embedding_1, embedding_2).numpy()[0]


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

    def preprocess(self):
        self.data = self.data.dropna()

    def save_data(self):
        self.data.to_csv('data/similarity_analysis_output.csv', index=False)


if __name__ == '__main__':
    sa = SimilarityAnalysis('data/intra_rater_data.csv')
    sa.calculate_intra_rater_similarity()
