import gensim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import os
from collections import Counter

# 设置日志以查看gensim加载模型的进度
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 1. 加载预训练的Word2Vec模型
def load_word2vec_model(model_path):
    print("加载Word2Vec模型...")
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("模型加载完成！")
    return model
# 2. 加载WordSim353数据集
def load_wordsim353(dataset_path):
    print("加载WordSim353数据集...")
    df = pd.read_csv(dataset_path, sep='\t', header=None, names=['word1', 'word2', 'human_score'])
    print("数据集加载完成！")
    return df

# 3. 计算词对的余弦相似度
def compute_cosine_similarity(model, word1, word2):
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

    # 计算Spearman相关系数
    correlation, p_value = spearmanr(similarities, human_scores)
    return correlation

# 5. 降维并评估不同维度下的性能
def evaluate_different_dimensions(model, df, dimensions=[100, 200, 300]):
    results = {}
    models = {}
    original_vectors = model.vectors
    original_dim = model.vector_size

    for dim in dimensions:
        if dim < original_dim:
            print(f"使用PCA将维度从{original_dim}降到{dim}...")
            pca = PCA(n_components=dim, random_state=42)
            reduced_vectors = pca.fit_transform(original_vectors)
            # 创建一个新的KeyedVectors实例
            reduced_model = gensim.models.KeyedVectors(vector_size=dim)
            reduced_model.add_vectors(model.index_to_key, reduced_vectors)
            print(f"维度降至{dim}完成！")
        elif dim == original_dim:
            reduced_model = model
            print(f"使用原始维度：{dim}")
        else:
            print(f"维度{dim}大于原始维度{original_dim}，跳过。")
            continue

        correlation = evaluate_spearman(reduced_model, df)
        results[dim] = correlation
        models[dim] = reduced_model
        print(f"维度{dim}下的Spearman相关系数: {correlation:.4f}")

    return results, models

# 6. 使用t-SNE进行可视化并保存图像
def visualize_embeddings(model, words, dimension, output_dir, label, dimensions=2, perplexity=30, random_state=42):
    print(f"提取维度{dimension}下的词向量 ({label})...")
    vectors = []
    valid_words = []
    for word in words:
        if word in model:
            vectors.append(model[word])
            valid_words.append(word)
    print(f"提取了{len(vectors)}个有效词的向量。")

    if len(vectors) == 0:
        print(f"没有有效的词汇用于可视化（维度{dimension}，标签{label}）。")
        return

    # 将列表转换为NumPy数组
    vectors = np.array(vectors)

    print("进行t-SNE降维...")
    tsne = TSNE(n_components=dimensions, perplexity=perplexity, random_state=random_state, init='random')
    vectors_tsne = tsne.fit_transform(vectors)
    print("t-SNE降维完成！")

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘图
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], marker='o', color='skyblue', edgecolors='k', alpha=0.7)

    for i, word in enumerate(valid_words):
        plt.annotate(word, (vectors_tsne[i, 0], vectors_tsne[i, 1]),
                     textcoords="offset points", xytext=(5, 2), ha='left', fontsize=9)

    plt.title(f't-SNE Visualization of Word Embeddings (Dimension: {dimension}, {label})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, f'tsne_dim{dimension}_{label}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE可视化图已保存至 {output_path}")

# 主函数
def main():
    # 文件路径（请根据实际情况修改）
    model_path = 'models/GoogleNews-vectors-negative300.bin'  # 预训练模型路径
    dataset_path = 'data/wordsim353/wordsim353.tsv'  # 数据集路径
    output_dir = 'tsne_visualizations/google'  # 保存可视化图的目录

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请确认路径是否正确。")
        return

    # 检查数据集文件是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集文件 {dataset_path} 不存在，请确认路径是否正确。")
        return

    # 下载必要的nltk数据（如果需要）
    # nltk.download('punkt')
    # nltk.download('wordnet')

    # 加载模型和数据集
    model = load_word2vec_model(model_path)
    df = load_wordsim353(dataset_path)

    # 评估不同维度下的Spearman相关系数，并获取相应的模型
    dimensions = [100, 200, 300]
    print("\n评估不同维度下的词嵌入性能...")
    spearman_results, models = evaluate_different_dimensions(model, df, dimensions)
    print("\n不同维度下的Spearman相关系数:")
    for dim, corr in spearman_results.items():
        print(f"维度 {dim}: Spearman相关系数 = {corr:.4f}")

    # 计算词频并选择频率前100的词
    print("\n计算词频并选择频率前100的词...")
    all_words = df['word1'].tolist() + df['word2'].tolist()
    word_freq = Counter(all_words)
    top_100_words = [word for word, freq in word_freq.most_common(100)]
    print(f"选择了频率前100的词用于可视化。")

    # 获取所有唯一的词
    all_unique_words = list(set(all_words))
    print(f"总共有{len(all_unique_words)}个唯一的词用于可视化。")

    # 可视化部分：为每个维度生成两种t-SNE图并保存
    print("\n准备进行t-SNE可视化并保存图像...")

    for dim in dimensions:
        if dim in models:
            # 可视化频率前100的词
            visualize_embeddings(models[dim], top_100_words, dim, output_dir, label='top100')

            # 可视化所有的词
            visualize_embeddings(models[dim], all_unique_words, dim, output_dir, label='all')
        else:
            print(f"维度{dim}的模型不存在，跳过可视化。")

    # 保存Spearman相关系数结果到文件
    spearman_output_path = os.path.join(output_dir, 'spearman/google_spearman_results.txt')
    with open(spearman_output_path, 'w') as f:
        for dim, corr in spearman_results.items():
            f.write(f"维度 {dim}: Spearman相关系数 = {corr:.4f}\n")
    print(f"\nSpearman相关系数结果已保存至 {spearman_output_path}")

    print("\n所有t-SNE可视化图已生成并保存。")

if __name__ == "__main__":
    main()
