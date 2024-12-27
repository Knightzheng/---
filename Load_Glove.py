import gensim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import os
from collections import Counter
from gensim.scripts.glove2word2vec import glove2word2vec

# 设置日志以查看gensim加载模型的进度
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def convert_glove_to_word2vec(glove_input_file, word2vec_output_file):
    if not os.path.exists(word2vec_output_file):
        glove2word2vec(glove_input_file, word2vec_output_file)
    else:
        print("已经转换完成，可以直接加载")
def load_glove_model(glove_file, word2vec_file):
    """
    加载GloVe模型，首先将其转换为Word2Vec格式，然后加载。
    """
    convert_glove_to_word2vec(glove_file, word2vec_file)
    print(f"加载GloVe模型 {glove_file}...")
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
    print(f"GloVe模型 {glove_file} 加载完成！")
    return model
def load_wordsim353(dataset_path):
    """
    加载WordSim353数据集。
    """
    print(f"加载WordSim353数据集 {dataset_path}...")
    df = pd.read_csv(dataset_path, sep='\t', header=None, names=['word1', 'word2', 'human_score'])
    print("WordSim353数据集加载完成！")
    return df
def compute_cosine_similarity(model, word1, word2):
    """
    计算两个词的余弦相似度。如果词不在模型中，则返回None。
    """
    try:
        return model.similarity(word1, word2)
    except KeyError:
        # 如果词汇不在词表中，则返回None
        return None
# 4. 评估Spearman相关系数
def evaluate_spearman(model, df):
    similarities = []
    human_scores = []
    missing = 0
    for index, row in df.iterrows():
        sim = compute_cosine_similarity(model, row['word1'], row['word2'])
        if sim is not None:
            similarities.append(sim)
            human_scores.append(row['human_score'])
        else:
            missing += 1
    if missing > 0:
        print(f"跳过了{missing}个不在词表中的词对。")
    if len(similarities) == 0:
        print("没有有效的词对用于计算Spearman相关系数。")
        return None
    correlation, p_value = spearmanr(similarities, human_scores)# 计算Spearman相关系数
    return correlation
# 5. 降维并评估不同维度下的性能
# 6. 使用t-SNE进行可视化并保存图像


#使用t-SNE进行降维并生成可视化图像。
def visualize_embeddings(model, words, dimension, output_dir, label, dimensions=2, perplexity=30, random_state=42):
    print(f"提取维度{dimension}下的词向量 ({label})...")
    vectors = []
    valid_words = []
    for word in words:
        if word in model:
            vectors.append(model[word])
            valid_words.append(word)
    if len(vectors) == 0:
        print(f"没有有效的词汇用于可视化（维度{dimension}，标签{label}）。")
        return
    vectors = np.array(vectors)  # 将列表转换为NumPy数组
    tsne = TSNE(n_components=dimensions, #进行tsne降维
                perplexity=perplexity, random_state=random_state, init='random')
    vectors_tsne = tsne.fit_transform(vectors)
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 绘图
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], marker='o',
                color='skyblue', edgecolors='k', alpha=0.7)
    for i, word in enumerate(valid_words):
        plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                     textcoords="offset points", xytext=(5, 2), ha='left', fontsize=9)
    plt.title(f't-SNE Visualization of Word Embeddings (Dimension: {dimension}, {label})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'tsne_dim{dimension}_{label}.png')# 保存图像
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE可视化图已保存至 {output_path}")

def main():
    # 文件路径（请根据实际情况修改）
    dataset_path = 'data/wordsim353/wordsim353.tsv'  # 数据集路径
    glove_files = {
        100: 'models/glove.6B.100d.txt',
        200: 'models/glove.6B.200d.txt',
        300: 'models/glove.6B.300d.txt'
    }
    output_dir = 'tsne_visualizations/glove'  # 保存可视化图的目录
    models_dir = 'models'  # 保存转换后的Word2Vec模型的目录

    # 创建models目录（如果不存在）
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"已创建模型保存目录：{models_dir}")
    else:
        print(f"模型保存目录 {models_dir} 已存在。")

    # 检查数据集文件是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集文件 {dataset_path} 不存在，请确认路径是否正确。")
        return

    # 加载WordSim353数据集
    df = load_wordsim353(dataset_path)

    # 计算词频并选择频率前100的词
    print("\n计算词频并选择频率前100的词...")
    all_words = df['word1'].tolist() + df['word2'].tolist()
    word_freq = Counter(all_words)
    top_100_words = [word for word, freq in word_freq.most_common(100)]
    print(f"选择了频率前100的词用于可视化。")

    # 获取所有唯一的词
    all_unique_words = list(set(all_words))
    print(f"总共有{len(all_unique_words)}个唯一的词用于可视化。")

    # 初始化一个字典来存储Spearman相关系数
    spearman_results = {}

    # 迭代加载每个GloVe模型并进行评估与可视化
    for dim, glove_file in glove_files.items():
        if not os.path.exists(glove_file):
            print(f"GloVe文件 {glove_file} 不存在，请确认路径是否正确。")
            continue

        # 定义转换后的Word2Vec格式文件名，并存储在models目录中
        word2vec_file = os.path.join(models_dir, f'glove.6B.{dim}d.word2vec.txt')

        # 加载GloVe模型
        model = load_glove_model(glove_file, word2vec_file)

        # 评估Spearman相关系数
        print(f"\n评估GloVe {dim}维度模型在WordSim353数据集上的Spearman相关系数...")
        correlation = evaluate_spearman(model, df)
        if correlation is not None:
            spearman_results[dim] = correlation
            print(f"维度 {dim}: Spearman相关系数 = {correlation:.4f}")
        else:
            spearman_results[dim] = 'N/A'
            print(f"维度 {dim}: Spearman相关系数 = N/A")

        # 可视化部分：生成并保存两种类型的t-SNE图
        print(f"\n准备为GloVe {dim}维度模型生成t-SNE可视化图像...")
        visualize_embeddings(model, top_100_words, dim, output_dir, label='top100')
        visualize_embeddings(model, all_unique_words, dim, output_dir, label='all')

    # 保存Spearman相关系数结果到文件，包含时间戳以避免覆盖
    spearman_output_path = os.path.join(output_dir, f'spearman_results.txt')
    with open(spearman_output_path, 'w') as f:
        f.write("Spearman相关系数结果:\n")
        for dim, corr in spearman_results.items():
            f.write(f"维度 {dim}: Spearman相关系数 = {corr}\n")
    print(f"\nSpearman相关系数结果已保存至 {spearman_output_path}")

    print("\n所有t-SNE可视化图已生成并保存。")

if __name__ == "__main__":
    main()
