import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Callable, List
from common_utils import config
from .process_data import ProcessData

# !/text_processing/
#
# dea.py
# Gabriel de Jesus (mestregabrieldejesus@gmail.com)
# 15-04-2023


class DataAnalysisExploration:
    """
    This class performs data analysis and exploration.
    """

    def __init__(self, dataset) -> None:
        self.process_data = ProcessData()
        self.dataset = dataset
        self.plots = config.plots
        self.data_frac_for_clustering = config.DATA_FRAC_FOR_CLUSTERING

    def count_sentences(self) -> List[int]:
        """ Counts each sentence length for each language and return a list of integer. """

        languages = list(set(self.dataset["language"]))
        data = [
            self.dataset[self.dataset["language"] == lang]["sentence_length"] for lang in languages
        ]

        return data

    def words_summary(self) -> pd.DataFrame:
        """ Summarize sentence and words in each sentence, including the overall total, 
        and return a DataFrame containing their summary. """

        summary = []
        for lang in list(set(self.dataset["language"])):
            sentence_list = self.dataset["sentence"][self.dataset["language"] == lang]
            words = sentence_list.str.split()
            words_count = words.apply(len)
            max_words_per_documents = words_count.max()
            min_words_per_documents = words_count.min()
            avg_words_per_documents = words_count.mean()
            total_words_in_doc = words_count.sum()

            summary.append(
                {
                    "language": lang,
                    "max_words/documents": max_words_per_documents,
                    "min_words/documents": min_words_per_documents,
                    "avg_words/documents": avg_words_per_documents,
                    "total_words_in_doc": total_words_in_doc,
                }
            )

        return pd.DataFrame(summary)

    def display_data_in_bar(self) -> None:
        """ Visualize dataset in a bar plot. """

        counts = self.dataset["language"].value_counts()
        counts.plot(kind="bar")
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha="center", va="bottom")
        plt.title("Total of sentences per language")
        plt.xlabel("Language")
        plt.ylabel("Total")
        plt.show()

    def display_data_in_boxplot(self) -> None:
        """ Visualize dataset for each language in boxplot and returns
        a plot contains 4 boxplots, one for each language. """

        languages = list(set(self.dataset["language"]))
        fig, ax = plt.subplots()
        ax.boxplot(self.count_sentences())
        ax.set_xticklabels(languages)
        plt.title("Sentences distribution by language")
        plt.xlabel("Languages")
        plt.ylabel("Length (sentences)")
        plt.show()

    def display_data_in_gaussian_dist(self, display_lines: bool = False) -> None:
        """ Counts each sentence length for each language.
        :param display_lines (bool): whether displaying lines or not (default=False)
        """

        languages = list(set(self.dataset["language"]))
        data_counts = self.count_sentences()

        num_rows = 2
        num_cols = 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        for i in range(len(data_counts)):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]

            # Outlier detection using the Standard Deviation Method.
            cut_off_1 = np.std(data_counts[i]) * 1
            mean_value = np.mean(data_counts[i])

            # Outlier detection wusingith the Inter-Quartile Method.
            q25 = np.percentile(data_counts[i], 25)
            q75 = np.percentile(data_counts[i], 75)
            iqr = q75 - q25
            cut_off = iqr * 1
            # lower, upper = iqr - cut_off, iqr + cut_off

            # Plot the gaussian distribution
            sns.kdeplot(data_counts[i], shade=True, ax=ax)
            if display_lines:
                ax.axvline(
                    mean_value - cut_off_1,
                    linestyle="dashed",
                    color="green",
                    label="lower boundary for std 1",
                )
                ax.axvline(
                    mean_value + cut_off_1,
                    linestyle="dashed",
                    color="green",
                    label="upper boundary for std 1",
                )
                ax.axvline(
                    iqr - cut_off, linestyle="dashed", color="red", label="lower boundary for IQR 1"
                )
                ax.axvline(
                    iqr + cut_off, linestyle="dashed", color="red", label="upper boundary for IQR 1"
                )
                ax.legend()
            ax.set_xlabel(languages[i])
            # ax.set_title("Sentence length distribution by language")

        # Add padding between subplots
        plt.tight_layout()
        plt.show()

    def display_data_in_clustering(
        self,
        algorithm: Callable,
        title: str,
        num_clusters: int = 4,
        using_random_state: bool = False,
        using_n_components: bool = False,
    ) -> None:
        """ 
        Visualize clustering of each language.

        :param algorithm: clustering algorithm name.
        :paramtitle (str): the graph title.
        :param num_clusters (int): number of clusters (4 default).
        :param num_components (bool): whether a n_components parameter is used (default=False).
        :param random_state (bool): whether a random_state parameter is used (default=False).
        """

        data = self.dataset.sample(
            frac=self.data_frac_for_clustering, random_state=42
        )  # Reduce the data size.
        input_tokens = data["token"]

        # Convert the text data into numerical vectors.
        vectorizer = CountVectorizer()
        words = vectorizer.fit_transform(input_tokens)

        # Reduce data dimensionality to 2 dimensions and normalize sentence length.
        pca = PCA(n_components=2)
        words_pca = pca.fit_transform(words.toarray())
        scaler = StandardScaler()
        norm_words = scaler.fit_transform(words_pca)

        unique_labels = list(set(data["language"]))
        colors = ["red", "blue", "green", "orange"]

        label_color_dict = {}
        for i, label in enumerate(unique_labels):
            label_color_dict[label] = colors[i % len(colors)]

        # Apply clustering algorithms.
        if not using_random_state:
            algorithm(n_clusters=num_clusters).fit(norm_words)
        else:
            if using_n_components:
                algorithm(n_components=num_clusters,
                          random_state=42).fit(norm_words)
            else:
                algorithm(n_clusters=num_clusters,
                          random_state=42).fit(norm_words)

        # Plot the clusters found with different colors for each label.
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, label in enumerate(data["language"]):
            ax.scatter(
                norm_words[i, 0], norm_words[i, 1], color=label_color_dict[label])

        legend_handles = []
        for label, color in label_color_dict.items():
            legend_handles.append(ax.scatter([], [], color=color, label=label))
        ax.legend(handles=legend_handles)
        ax.set_xlabel(title)

        plt.savefig(str(self.plots / (title.lower() + ".png")))
        plt.close(fig)

    def display_images(self, *args: str) -> None:
        """ Visualize images in a plot. """

        img1 = Image.open(os.path.join(self.plots, args[0])) if len(
            args) >= 1 else None
        img2 = Image.open(os.path.join(self.plots, args[1])) if len(
            args) >= 2 else None
        img3 = Image.open(os.path.join(self.plots, args[2])) if len(
            args) >= 3 else None
        img4 = Image.open(os.path.join(self.plots, args[3])) if len(
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
