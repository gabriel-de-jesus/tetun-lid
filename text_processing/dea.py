import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
from typing import Callable, List


class DataAnalysisExploration:
    """
    A class for performing data analysis and exploration.
    """

    def count_sentences(self, data: pd.DataFrame) -> List[int]:
        """ Counts each sentence chars length for each language.

        Args:
            data (DataFrame): A DataFrame contained the preprocessed data.

        Returns:
            A list contains list of Tetun (tet), Portuguese (pt), English (en),
            and Indonesian (id) language
        """

        languages = list(set(data['language']))
        all_language_scores = []
        for language in languages:
            language_data = data[data['language'] == language]
            score_sentences = [len(s) for s in language_data['sentence']]
            all_language_scores.append(score_sentences)

        return all_language_scores

    def display_data_in_bar(self, data: pd.DataFrame) -> None:
        """ Visualize dataset in a bar plot.

        Args:
            data (DataFrame): a DataFrame contained a preprocessed data.

        Results:
            A bar plot illustrates the total sentences for each language.
        """

        counts = data['language'].value_counts()
        counts.plot(kind='bar')
        for i, count in enumerate(counts):
            plt.text(i, count+0.5, str(count), ha='center', va='bottom')
        plt.title("Total of sentences per language")
        plt.xlabel("Language")
        plt.ylabel("Total")
        plt.show()

    def display_data_in_boxplot(self, data: pd.DataFrame) -> None:
        """ Visualize dataset for each language in boxplot.

        Args:
            data (DataFrame): a DataFrame contained a preprocessed data.

        Results:
            A plot contains 4 boxplots, one for each language
        """

        languages = list(set(data['language']))

        # Plot in the boxplots
        fig, ax = plt.subplots()
        ax.boxplot(self.count_sentences(data))
        ax.set_xticklabels(languages)
        plt.title("Sentences distribution by language")
        plt.xlabel('Languages')
        plt.ylabel('Length (sentences)')
        plt.show()

    def display_data_in_gaussian_dist(
            self,
            data: pd.DataFrame,
            count_sentences: List[int] = count_sentences,
            display_lines: bool = False
    ) -> None:
        """ Counts each sentence length for each language.

        Args:
            data (DataFrame): a DataFrame contained a preprocessed data.
            sentece_counts (list): a list of language lists.
            display_lines (bool): whether displaying lines or not (default=False)

        Results:
            A plot contains 4 subplots, one for each language
        """

        # Get a list of the language
        languages = list(set(data['language']))

        # A list that contains lists of the sentences length for each language
        data_counts = self.count_sentences(data)

        num_rows = 2
        num_cols = 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        for i in range(len(data_counts)):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]

            # Outlier detection with the Standard Deviation Method
            cut_off_1 = np.std(data_counts[i]) * 1
            # cut_off_2 = np.std(data_counts[i]) * 2
            # cut_off_3 = np.std(data_counts[i]) * 3
            mean_value = np.mean(data_counts[i])

            # Outlier detection with the Inter-Quartil Method
            q25 = np.percentile(data_counts[i], 25)
            q75 = np.percentile(data_counts[i], 75)
            iqr = q75 - q25
            cut_off = iqr * 1
            # lower, upper = iqr - cut_off, iqr + cut_off

            # Plot the gaussian distribution
            sns.kdeplot(data_counts[i], shade=True, ax=ax)
            if display_lines:
                ax.axvline(mean_value - cut_off_1, linestyle='dashed',
                           color='green', label='lower boundary for std 1')
                ax.axvline(mean_value + cut_off_1, linestyle='dashed',
                           color='green', label='upper boundary for std 1')
                ax.axvline(iqr - cut_off, linestyle='dashed',
                           color='red', label='lower boundary for IQR 1')
                ax.axvline(iqr + cut_off, linestyle='dashed',
                           color='red', label='upper boundary for IQR 1')
                ax.legend()
            ax.set_xlabel(languages[i])
            # ax.set_title("Sentence length distribution by language")

        # Add padding between subplots
        plt.tight_layout()
        plt.show()

    def display_data_in_clustering(
            self,
            data: pd.DataFrame,
            algorithm: Callable,
            title: str,
            num_clusters: int = 4,
            using_random_state: bool = False,
            using_n_components: bool = False
    ) -> None:
        """ Visualize clustering of each language.

        Args:
            data (DataFrame): a DataFrame contained a preprocessed data.
            algorithm: clustering algorithm name.
            title (str): the graph title.
            um_clusters (int): number of clusters (4 default).
            num_components (bool): whether a n_components parameter is used (default=False).
            random_state (bool): whether a random_state parameter is used (default=False).

        Results:
            A plot of 2 dimensional.
        """
        input_sentences = data['sentence']

        # Convert the text data into numerical vectors
        vectorizer = CountVectorizer()
        sentences = vectorizer.fit_transform(input_sentences)

        # Reduce data dimensionality to 2 dimensions
        pca = PCA(n_components=2)
        sentences_pca = pca.fit_transform(sentences.toarray())

        # Normalize the sentence length
        scaler = StandardScaler().fit(sentences_pca)
        norm_sentences = scaler.transform(sentences_pca)

        # Get unique labels
        unique_labels = list(set(data['language']))

        # Define a list of colors
        colors = ['red', 'blue', 'green', 'orange']

        # Assign a color to each unique label
        label_color_dict = {}
        for i, label in enumerate(unique_labels):
            label_color_dict[label] = colors[i % len(colors)]

        # Apply clustering algorithm with k=4
        if not using_random_state:
            algorithm(n_clusters=num_clusters).fit(norm_sentences)
        else:
            if using_n_components:
                algorithm(n_components=num_clusters,
                          random_state=42).fit(norm_sentences)
            else:
                algorithm(n_clusters=num_clusters,
                          random_state=42).fit(norm_sentences)

        # Plot the clusters found with different colors for each label
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, label in enumerate(data['language']):
            ax.scatter(norm_sentences[i, 0], norm_sentences[i,
                                                            1], color=label_color_dict[label])

        # Add legend
        legend_handles = []
        for label, color in label_color_dict.items():
            legend_handles.append(ax.scatter([], [], color=color, label=label))
        ax.legend(handles=legend_handles)

        ax.set_xlabel(title)

        plt.savefig('plots/'+title.lower()+'.png')
        plt.close(fig)

    def display_images(self, *args: str) -> None:
        """Visualize images in a plot.

        Args:
            *args (str): image names (for 4 images).

        Results:
            A plot contains 4 plots, one for each image.
        """
        img_dir = "plots/"

        img1 = Image.open(os.path.join(img_dir, args[0])) if len(
            args) >= 1 else None
        img2 = Image.open(os.path.join(img_dir, args[1])) if len(
            args) >= 2 else None
        img3 = Image.open(os.path.join(img_dir, args[2])) if len(
            args) >= 3 else None
        img4 = Image.open(os.path.join(img_dir, args[3])) if len(
            args) >= 4 else None

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs[0, 0].imshow(img1) if img1 else None
        axs[0, 1].imshow(img2) if img2 else None
        axs[1, 0].imshow(img3) if img3 else None
        axs[1, 1].imshow(img4) if img4 else None

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        plt.show()
