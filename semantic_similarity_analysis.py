import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from tqdm import tqdm

tqdm.pandas()


class SemanticSimilarityAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def preprocess(self):
        self.data = self.data.dropna()

    def calculate_sentence_sim(self, row):
        embedding_1 = self.model.encode(row['feedback_1'], convert_to_tensor=True).cpu()
        embedding_2 = self.model.encode(row['feedback_2'], convert_to_tensor=True).cpu()

        return util.pytorch_cos_sim(embedding_1, embedding_2).detach().numpy()[0][0]


class InterRaterSemanticSimilarityAnalysis(SemanticSimilarityAnalysis):

    def __init__(self, data_path):
        super().__init__(data_path)
        self.output_data = None

    def calculate_similarity(self):
        print("Preprocessing data...")
        self.preprocess()
        print("Formatting table for inter_rater")
        self.process_inter_rater()
        print("Calculating pair-wise similarity...")
        self.output_data['sim'] = self.output_data.progress_apply(self.calculate_sentence_sim, axis=1)
        print("Saving....")

        self.output_data.to_csv('data/inter_rater_sim.csv', index=False)
        
    def process_inter_rater(self):
        for group in tqdm(self.data['group'].unique()):
            for skill in self.data['skill'].unique():
                for n in self.data[(self.data['group'] == group) &
                                   (self.data['skill'] == skill)]['assignment_number'].unique():
                    if self.output_data is None:
                        self.output_data = self.generate_feedback_pairs(group, skill, n)
                    else:
                        self.output_data = pd.concat([self.output_data, self.generate_feedback_pairs(group, skill, n)])

        print('\n Removing empty feedback pairs and filling single missing comments...')
        self.output_data = self.output_data.dropna(how='all')
        self.output_data = self.output_data.fillna('No comment supplied')

    def generate_feedback_pairs(self, group, skill, assignment_number):
        group_bounds = group.split('-')
        pairs = combinations(range(int(group_bounds[0]), int(group_bounds[1]) + 1), r=2)

        feedback_pairs_df = None

        for idx, pair in enumerate(pairs):
            feedback_1 = self.get_comments(group, skill, assignment_number, pair[0])
            feedback_2 = self.get_comments(group, skill,assignment_number, pair[1])

            if feedback_pairs_df is None:
                feedback_pairs_df = pd.DataFrame({'group': group,
                                                  'skill': skill,
                                                  'assignment_number': assignment_number,
                                                  'participant_id_1': pair[0],
                                                  'participant_id_2': pair[1],
                                                  'feedback_1': feedback_1,
                                                  'feedback_2': feedback_2
                                                  }, index=[idx])
            else:
                feedback_pairs_df = pd.concat([
                    feedback_pairs_df,
                    pd.DataFrame({'group': group, 'assignment_number': assignment_number, 'skill': skill,
                                  'participant_id_1': pair[0],
                                  'participant_id_2': pair[1],
                                  'feedback_1': feedback_1,
                                  'feedback_2': feedback_2
                                  }, index=[idx])]
                )

        return feedback_pairs_df

    def get_comments(self, group, skill, assignment_number, participant_id):
        feedback = self.data[(self.data['group'] == group) &
                        (self.data['skill'] == skill) &
                        (self.data['assignment_number'] == assignment_number) &
                        (self.data['participant_id'] == participant_id)]['comments']

        if feedback.empty:
            return None
        return feedback.reset_index(drop=True)[0]


class IntraRaterSemanticSimilarityAnalysis(SemanticSimilarityAnalysis):
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
        self.data = self.data.progress_apply(self.calculate_sentence_sim, axis=1)
        print("Saving....")
        self.data.to_csv('data/intra_rater_sim.csv')

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
    parser = argparse.ArgumentParser(description='Run Inter/Intra rater sentence semantic similarity analysis')
    parser.add_argument('-inter', dest='inter', default=False, help='Run in inter-rater analysis',
                        action='store_true')
    parser.add_argument('-intra', dest='intra', default=False, help='Run in intra-rater analysis',
                        action='store_true')

    args = parser.parse_args()

    if args.inter:
        sa = InterRaterSemanticSimilarityAnalysis('data/inter_rater.csv')
        sa.calculate_similarity()
    elif args.intra:
        sa = IntraRaterSemanticSimilarityAnalysis('data/intra_rater.csv')
        sa.calculate_similarity()
    else:
        print('Please select -inter or -intra to run the semantic similarity analysis')



